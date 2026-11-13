// fireredsim platform layer: retail hardware-fault emulation (docs/ROBUSTNESS.md).
//
// Retail FireRed reads and writes through NULL and stale pointers (VBlank
// callbacks that outlive the data they use, NULL templates, ...), and could
// divide by zero. On the GBA these have defined results:
//
//   * reads of the BIOS region 0x00000000-0x00003FFF while code runs outside
//     the BIOS return the BIOS "open bus": the last opcode the CPU prefetched
//     from the BIOS (mGBA: memory.biosPrefetch; gPortBiosPrefetch here,
//     0xE3A02004 after a SWI, 0xE25EF004 inside the IRQ handler, 0xE55EC002
//     after it). mGBA's GBALoad8/16/32 return (p >> 8*(a&3)) & 0xFF,
//     (p >> 8*(a&2)) & 0xFFFF and ROR(p, 8*(a&3)) for a byte, halfword and word
//     load at address a: exactly what loads from memory filled with the word p
//     repeated (little endian) return, for every aligned access and for
//     unaligned words (the ARM rotation equals the x86 unaligned read of a
//     4-periodic pattern). Only a halfword load at an odd address differs (ARM
//     rotates); such an access is reported as inexact.
//   * writes to the BIOS region and to unmapped memory are ignored;
//   * EWRAM, IWRAM, palette, VRAM, OAM and the flash window are mirrored over
//     their 16 MiB areas;
//   * agbcc's libgcc returns 0 for x / 0 and x % 0, and INT_MIN (remainder 0)
//     for INT_MIN / -1.
//
// Natively those accesses fault. This module catches SIGSEGV/SIGBUS/SIGFPE/
// SIGILL raised by game code (the stack pointer is on the game stack,
// platform/frame.c) on an alternate signal stack and:
//
//   * memory operand (ModRM/moffs): decodes the instruction (x86_decode.c),
//     computes the GBA address it accessed, and re-executes the instruction
//     unchanged from a trampoline with only its memory operand replaced by an
//     absolute address: a shadow copy of the BIOS region filled with the
//     current open-bus word (refreshed before every redirected access, so
//     writes into it are discarded), a scratch area for ignored writes, or the
//     mirrored native address. Registers, flags and every other effect are the
//     CPU's own, so read-modify-write, x87 and SSE instructions are covered
//     without emulating them. Zero cost when nothing faults.
//   * string instructions (rep movs/stos/lods): emulated element by element
//     through the same GBA bus rules.
//   * divide error: the agbcc libgcc result is written to EAX/EDX and the
//     instruction skipped (only for 32-bit C divisions: EDX:EAX holds the
//     sign/zero extension of EAX).
//   * anything that cannot be reproduced exactly (a read of unmapped memory =
//     CPU open bus, which is the prefetched game opcode; a call through a
//     pointer read from unmapped memory; an illegal instruction; a game stack
//     overflow; the hang watchdog) ends the environment with a terminal status
//     (frame.c) instead of the process.
//
// Every distinct site (instruction address + kind) is counted and reported
// once per process. Faults raised by host code (not on the game stack) are
// passed to the handler that was installed before (sim's diagnostics).
//
// This file includes system headers, so the ARM struct-layout macro of the
// force-included gba/portable_abi.h is switched off here; it uses no game
// header.
#undef struct
#undef union

#include <stddef.h>
#include <stdint.h>
#include <signal.h>
#include <ucontext.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <pthread.h>
#include <elf.h>

#include "fireredsim_port.h"
#include "port_fault.h"
#include "x86_decode.h"

#define HOST_STATE __attribute__((section(".host.bss")))

#ifndef MAP_FIXED_NOREPLACE
#define MAP_FIXED_NOREPLACE 0x100000
#endif

// i386 ucontext gregs indices (sys/ucontext.h; spelled out because the
// REG_* names need _GNU_SOURCE, which the force-included headers precede).
enum {
    G_GS = 0, G_FS, G_ES, G_DS, G_EDI, G_ESI, G_EBP, G_ESP, G_EBX, G_EDX, G_ECX, G_EAX,
    G_TRAPNO, G_ERR, G_EIP, G_CS, G_EFL, G_UESP, G_SS,
};
// x86 register number (EAX=0 ... EDI=7) -> gregs index.
#define GREG(r) (G_EAX - (int)(r))

#define EFL_TF 0x100u
#define EFL_DF 0x400u

extern char __executable_start[];
extern char etext[];
extern unsigned char __pnstate_start[], __pnstate_end[];

// ---- host state ------------------------------------------------------------------

#define SHADOW_SIZE 0x20000u
#define SHADOW_BIOS 0x0u      // [0, 0x4000 + 64): BIOS open-bus pattern
#define SHADOW_SCRATCH 0x8000u // ignored writes
#define SHADOW_ZERO 0xC000u    // reads that return 0 (VRAM hole in bitmap modes)
#define BIOS_SIZE 0x4000u

HOST_STATE static int sInstalled;
HOST_STATE static int sInstallFailed;
HOST_STATE static uint8_t *sShadow;
HOST_STATE static uint8_t *sTramp;
HOST_STATE static uint32_t sImageEnd; // end of the executable's mapped image (ROM window)
HOST_STATE static uint32_t sImageRwStart, sImageRwEnd; // its writable segment (host data)
HOST_STATE static struct sigaction sOldSegv, sOldBus, sOldFpe, sOldIll, sOldWatch;
HOST_STATE static pthread_t sPreparedThread;
HOST_STATE static int sPreparedValid;
HOST_STATE static __attribute__((aligned(16))) uint8_t sRescueStack[16384];

// Hang watchdog (Port_SetHangWatchdog): 0 = default, 0xFFFFFFFF = off.
#define WATCHDOG_DEFAULT_MS 2000u
HOST_STATE static uint32_t sWatchdogCfg;
HOST_STATE static int sWatchdogTimerValid;
HOST_STATE static timer_t sWatchdogTimer;
HOST_STATE static volatile int sWatchArmed;
HOST_STATE static volatile uint32_t sWatchSeq;

// Fault sites.
#define MAX_SITES 1024
struct Site
{
    uint32_t eip, kind, addr, value;
    uint64_t count;
    uint32_t reported;
    uint32_t bt[PORT_BACKTRACE_MAX];
};
HOST_STATE static struct Site sSites[MAX_SITES];
HOST_STATE static volatile int sNumSites;
HOST_STATE static volatile int sReportPending;
HOST_STATE static volatile uint64_t sSitesDropped;
HOST_STATE static PortFaultReportFn sReportFn;
HOST_STATE static int sReportQuiet;

// ---- small helpers (no C library: they run in signal handlers) ---------------------

struct Out
{
    char *p;
    uint32_t n, cap;
};

static void PutC(struct Out *o, char c)
{
    if (o->n + 1 < o->cap)
        o->p[o->n++] = c;
    o->p[o->n] = 0;
}

static void PutS(struct Out *o, const char *s)
{
    while (s && *s)
        PutC(o, *s++);
}

static void PutHex(struct Out *o, uint32_t v, int digits)
{
    static const char kHex[] = "0123456789abcdef";
    int i;
    PutS(o, "0x");
    for (i = digits - 1; i >= 0; i--)
        PutC(o, kHex[(v >> (4 * i)) & 15]);
}

static void Fill32(uint8_t *dst, uint32_t value, uint32_t bytes)
{
    uint32_t i;
    for (i = 0; i + 4 <= bytes; i += 4)
        *(volatile uint32_t *)(dst + i) = value;
}

static int InText(uint32_t a)
{
    return a >= (uint32_t)__executable_start && a + 16 <= (uint32_t)etext;
}

static int InTrampoline(uint32_t a)
{
    return sTramp && a >= (uint32_t)sTramp && a < (uint32_t)sTramp + 4096;
}

static uint32_t StackLo(void)
{
    return (uint32_t)gPortGameStack;
}

static uint32_t StackHi(void)
{
    return (uint32_t)gPortGameStack + PORT_GAME_STACK_SIZE;
}

// True if the interrupted context runs game code: the game stack is in use and
// the stack pointer is on it (or just below it: an overflow).
static int IsGameContext(uint32_t esp)
{
    return gPortOnGameStack && esp >= StackLo() - 0x10000 && esp <= StackHi();
}

// ---- fault sites -------------------------------------------------------------------

static struct Site *FindSite(uint32_t eip, uint32_t kind)
{
    int i, n = sNumSites;
    for (i = 0; i < n; i++)
        if (sSites[i].eip == eip && sSites[i].kind == kind)
            return &sSites[i];
    return 0;
}

// Heuristic return-address scan of the game stack (no frame pointers in game
// code): a word is a return address if it points into the executable's code
// just after a CALL rel32 (E8) or an indirect CALL (FF /2) encoding.
static int IsReturnAddress(uint32_t w)
{
    const uint8_t *p = (const uint8_t *)w;
    if (w < (uint32_t)__executable_start + 8 || w >= (uint32_t)etext)
        return 0;
    if (p[-5] == 0xE8)
        return 1;
    if (p[-2] == 0xFF && (p[-1] & 0xF8) == 0xD0)
        return 1; // call *%reg
    if (p[-3] == 0xFF && (p[-2] & 0xF8) == 0x50)
        return 1; // call *disp8(%reg)
    if (p[-6] == 0xFF && ((p[-5] & 0xF8) == 0x90 || p[-5] == 0x15))
        return 1; // call *disp32(%reg) / *abs32
    if (p[-2] == 0xFF && (p[-1] & 0xF8) == 0x10 && (p[-1] & 7) != 4 && (p[-1] & 7) != 5)
        return 1; // call *(%reg)
    return 0;
}

int Port_GameBacktrace(uint32_t esp, uint32_t *out, int max)
{
    int n = 0;
    uint32_t a;
    if (esp < StackLo())
        esp = StackLo();
    for (a = (esp + 3) & ~3u; a + 4 <= StackHi() && n < max; a += 4)
    {
        uint32_t w = *(const uint32_t *)a;
        if (IsReturnAddress(w))
            out[n++] = w;
    }
    return n;
}

static void RecordSiteBt(uint32_t eip, uint32_t kind, uint32_t addr, uint32_t value, uint32_t esp)
{
    struct Site *s = FindSite(eip, kind);
    if (!s)
    {
        uint32_t i;
        if (sNumSites >= MAX_SITES)
        {
            sSitesDropped++;
            return;
        }
        s = &sSites[sNumSites];
        s->eip = eip;
        s->kind = kind;
        s->addr = addr;
        s->value = value;
        s->count = 0;
        s->reported = 0;
        for (i = 0; i < PORT_BACKTRACE_MAX; i++)
            s->bt[i] = 0;
        if (esp)
            Port_GameBacktrace(esp, s->bt, PORT_BACKTRACE_MAX);
        sNumSites++;
        sReportPending = 1;
    }
    s->count++;
}

// Site bookkeeping requested by game code (explicit notes) runs on a private
// host stack: on the game stack (environment state) a site's first occurrence
// would leave different bytes below the stack pointer than later occurrences,
// making the state depend on the process's history (found by
// fireredsim-fuzz --determinism).
HOST_STATE static __attribute__((aligned(16))) uint8_t sRecordStack[16384];
HOST_STATE static uint32_t sRecArgs[5];

void Port_CallOnStack(void (*fn)(void), void *stackTop);
__asm__(
    "    .text\n"
    "    .p2align 4\n"
    "    .globl Port_CallOnStack\n"
    "    .type Port_CallOnStack, @function\n"
    "Port_CallOnStack:\n"
    "    movl 4(%esp), %eax\n"   // fn
    "    movl 8(%esp), %ecx\n"   // stackTop (16-byte aligned)
    "    movl %esp, %edx\n"
    "    movl %ecx, %esp\n"
    "    pushl %edx\n"           // the caller's stack pointer
    "    subl $12, %esp\n"       // 16-byte alignment at the call
    "    call *%eax\n"
    "    addl $12, %esp\n"
    "    popl %esp\n"
    "    ret\n"
    "    .size Port_CallOnStack, .-Port_CallOnStack\n");

static void RecordFromArgs(void)
{
    RecordSiteBt(sRecArgs[0], sRecArgs[1], sRecArgs[2], sRecArgs[3], sRecArgs[4]);
}

static void RecordExplicit(uint32_t eip, uint32_t kind, uint32_t addr, uint32_t value, uint32_t esp)
{
    sRecArgs[0] = eip;
    sRecArgs[1] = kind;
    sRecArgs[2] = addr;
    sRecArgs[3] = value;
    sRecArgs[4] = esp;
    Port_CallOnStack(RecordFromArgs, sRecordStack + sizeof(sRecordStack));
}

void Port_FaultNoteExplicit(uint32_t kind, uint32_t addr, uint32_t value, const void *caller)
{
    RecordExplicit((uint32_t)caller, kind, addr, value, (uint32_t)__builtin_frame_address(0));
}

const char *Port_FaultKindName(uint32_t kind)
{
    switch (kind)
    {
    case PN_SITE_BIOS_READ: return "BIOS-region read (BIOS open bus)";
    case PN_SITE_BIOS_WRITE: return "BIOS-region write (ignored)";
    case PN_SITE_UNMAPPED_WRITE: return "write to unmapped memory (ignored)";
    case PN_SITE_MIRROR: return "mirrored memory access";
    case PN_SITE_ROM_WRITE: return "write to the cartridge ROM (ignored)";
    case PN_SITE_DIV_ZERO: return "division by zero (agbcc result 0)";
    case PN_SITE_DIV_OVERFLOW: return "INT_MIN / -1 (agbcc result INT_MIN)";
    case PN_SITE_OPEN_BUS: return "read of unmapped memory (CPU open bus: inexact, terminal)";
    case PN_SITE_TERMINAL: return "fault that ended the environment";
    case PN_SITE_EXPLICIT_BIOS: return "BIOS open bus requested by a PORTABLE patch";
    case PN_SITE_EXPLICIT_OPEN_BUS: return "open-bus load with the retail opcode, by a PORTABLE patch (exact)";
    case PN_SITE_SPIN_WAIT: return "busy-wait on the DMA3 queue inside a frame (VBlank delivered in the body, as the hardware lags)";
    }
    return "?";
}

// ---- symbolizer (host side, after the frame) ---------------------------------------

HOST_STATE static int sSymState; // 0 = not loaded, 1 = loaded, -1 = unavailable
HOST_STATE static const uint8_t *sSymFile;
HOST_STATE static const Elf32_Sym *sSyms;
HOST_STATE static uint32_t sNumSyms;
HOST_STATE static const char *sSymStr;
HOST_STATE static uint32_t sSymStrSize;

static void LoadSymbols(void)
{
    int fd;
    struct stat st;
    const Elf32_Ehdr *eh;
    const Elf32_Shdr *sh;
    uint32_t i;

    sSymState = -1;
    fd = open("/proc/self/exe", O_RDONLY);
    if (fd < 0)
        return;
    if (fstat(fd, &st) != 0 || st.st_size < (off_t)sizeof(Elf32_Ehdr))
    {
        close(fd);
        return;
    }
    sSymFile = mmap(0, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (sSymFile == MAP_FAILED)
    {
        sSymFile = 0;
        return;
    }
    eh = (const Elf32_Ehdr *)sSymFile;
    if (eh->e_ident[0] != 0x7F || eh->e_ident[1] != 'E' || eh->e_shoff == 0 ||
        eh->e_shoff + (uint32_t)eh->e_shnum * sizeof(Elf32_Shdr) > (uint32_t)st.st_size)
        return;
    sh = (const Elf32_Shdr *)(sSymFile + eh->e_shoff);
    for (i = 0; i < eh->e_shnum; i++)
    {
        if (sh[i].sh_type == SHT_SYMTAB && sh[i].sh_link < eh->e_shnum)
        {
            sSyms = (const Elf32_Sym *)(sSymFile + sh[i].sh_offset);
            sNumSyms = sh[i].sh_size / sizeof(Elf32_Sym);
            sSymStr = (const char *)(sSymFile + sh[sh[i].sh_link].sh_offset);
            sSymStrSize = sh[sh[i].sh_link].sh_size;
            sSymState = 1;
            return;
        }
    }
}

static void Symbolize(uint32_t addr, char *buf, uint32_t size)
{
    struct Out o = {buf, 0, size};
    const Elf32_Sym *best = 0;
    uint32_t i;

    buf[0] = 0;
    if (sSymState == 0)
        LoadSymbols();
    if (sSymState == 1)
    {
        for (i = 0; i < sNumSyms; i++)
        {
            const Elf32_Sym *y = &sSyms[i];
            uint32_t type = ELF32_ST_TYPE(y->st_info);
            if (type != STT_FUNC && type != STT_NOTYPE)
                continue;
            if (y->st_value > addr || y->st_name == 0 || y->st_name >= sSymStrSize)
                continue;
            if (y->st_size ? addr < y->st_value + y->st_size : (!best || y->st_value > best->st_value))
            {
                if (y->st_size && (!best || !best->st_size || y->st_value >= best->st_value))
                    best = y;
                else if (!y->st_size && (!best || (!best->st_size && y->st_value > best->st_value)))
                    best = y;
            }
        }
    }
    if (best)
    {
        PutS(&o, sSymStr + best->st_name);
        PutS(&o, "+");
        PutHex(&o, addr - best->st_value, addr - best->st_value > 0xFFFF ? 8 : 2);
    }
    else
    {
        PutHex(&o, addr, 8);
    }
}

static void FillSite(const struct Site *s, struct PortFaultSite *out)
{
    out->eip = s->eip;
    out->kind = s->kind;
    out->addr = s->addr;
    out->value = s->value;
    out->count = s->count;
    for (int i = 0; i < PORT_BACKTRACE_MAX; i++)
        out->backtrace[i] = s->bt[i];
    Symbolize(s->eip, out->where, sizeof(out->where));
}

void Port_Symbolize(uint32_t addr, char *buf, uint32_t size)
{
    Symbolize(addr, buf, size);
}

static void PutBacktrace(struct Out *o, const uint32_t *bt)
{
    int i;
    char where[96];
    for (i = 0; i < PORT_BACKTRACE_MAX && bt[i]; i++)
    {
        Symbolize(bt[i], where, sizeof(where));
        PutS(o, i ? " <- " : "; called from ");
        PutS(o, where);
    }
}

int Port_FaultSiteCount(void)
{
    return sNumSites;
}

int Port_GetFaultSites(struct PortFaultSite *out, int max)
{
    int i, n = sNumSites;
    for (i = 0; i < n && i < max; i++)
        FillSite(&sSites[i], &out[i]);
    return n;
}

void Port_SetFaultReport(PortFaultReportFn fn, int mode)
{
    sReportFn = fn;
    sReportQuiet = mode == 0;
}

static void WriteErr(const char *s)
{
    uint32_t n = 0;
    while (s[n])
        n++;
    while (n)
    {
        ssize_t w = write(2, s, n);
        if (w <= 0)
            return;
        s += w;
        n -= (uint32_t)w;
    }
}

void Port_FaultFlushReports(void)
{
    if (gPortTerminalPending)
    {
        struct PortTerminal t;
        gPortTerminalPending = 0;
        if (Port_GetTerminal(&t) && !sReportQuiet)
        {
            char line[1024], where[96];
            struct Out o = {line, 0, sizeof(line)};
            PutS(&o, "fireredsim: environment ended: ");
            PutS(&o, Port_TerminalKindName(t.kind));
            PutS(&o, ": ");
            PutS(&o, t.message);
            if (t.eip)
            {
                Symbolize(t.eip, where, sizeof(where));
                PutS(&o, " [at ");
                PutS(&o, where);
                PutBacktrace(&o, t.backtrace);
                PutS(&o, "]");
            }
            PutS(&o, "\n");
            WriteErr(line);
        }
    }
    if (!sReportPending)
        return;
    sReportPending = 0;
    {
        int i, n = sNumSites;
        for (i = 0; i < n; i++)
        {
            struct Site *s = &sSites[i];
            struct PortFaultSite site;
            if (s->reported)
                continue;
            s->reported = 1;
            FillSite(s, &site);
            if (sReportFn)
            {
                sReportFn(&site);
            }
            else if (!sReportQuiet)
            {
                char line[1024];
                struct Out o = {line, 0, sizeof(line)};
                PutS(&o, "fireredsim: retail ");
                PutS(&o, Port_FaultKindName(site.kind));
                PutS(&o, " at ");
                PutS(&o, site.where);
                PutS(&o, " (eip ");
                PutHex(&o, site.eip, 8);
                PutS(&o, "), GBA address ");
                PutHex(&o, site.addr, 8);
                if (site.kind == PN_SITE_BIOS_READ || site.kind == PN_SITE_EXPLICIT_BIOS ||
                    site.kind == PN_SITE_EXPLICIT_OPEN_BUS)
                {
                    PutS(&o, ", open-bus word ");
                    PutHex(&o, site.value, 8);
                }
                if (site.kind == PN_SITE_SPIN_WAIT || site.kind == PN_SITE_OPEN_BUS || site.kind == PN_SITE_TERMINAL)
                    PutBacktrace(&o, site.backtrace);
                PutS(&o, "; reported once per site\n");
                WriteErr(line);
            }
        }
    }
}

// ---- GBA bus model ---------------------------------------------------------------------

enum {
    ROUTE_OK = 0,      // *target is where the access goes (native memory)
    ROUTE_INEXACT = 1, // read of unmapped memory: CPU open bus
    ROUTE_INTERNAL = 2,
};

struct Route
{
    uint32_t target;   // native address the access is redirected to
    uint32_t kind;     // PN_SITE_*
    const char *why;   // for terminal statuses
};

static uint32_t BiosWord(void)
{
    return gPortBiosPrefetch;
}

// Refreshes the BIOS shadow around [addr, addr + size) (GBA offsets).
static void RefreshBios(uint32_t addr, uint32_t size)
{
    uint32_t lo = addr >= 64 ? (addr - 64) & ~3u : 0;
    uint32_t hi = ((addr + size + 64) + 3) & ~3u;
    if (hi > BIOS_SIZE + 64)
        hi = BIOS_SIZE + 64;
    Fill32(sShadow + SHADOW_BIOS + lo, BiosWord(), hi - lo);
}

static int NativelyMapped(uint32_t a, uint32_t size)
{
    uint32_t end = a + size;
    if (end < a)
        return 0;
#define IN(base, len) (a >= (base) && end <= (base) + (len))
    return IN(0x02000000u, 0x40000u) || IN(0x03000000u, 0x8000u) || IN(0x04000000u, 0x400u) ||
           IN(0x05000000u, 0x400u) || IN(0x06000000u, 0x18000u) || IN(0x07000000u, 0x400u) ||
           IN(0x0E000000u, 0x20000u) ||
           (a >= (uint32_t)__pnstate_start && end <= (uint32_t)__pnstate_end);
#undef IN
}

static int MirrorTarget(uint32_t base, uint32_t mask, uint32_t a, uint32_t size, struct Route *r)
{
    uint32_t off = a & mask;
    if (off + size > mask + 1)
    {
        r->why = "an access crosses the end of a mirrored memory region";
        return ROUTE_INTERNAL;
    }
    r->target = base + off;
    r->kind = PN_SITE_MIRROR;
    return ROUTE_OK;
}

// Where an access of `size` bytes at GBA address `a` goes, for an access that
// reads (`reads`) and/or writes. Only called for addresses that faulted.
static int RouteAccess(uint32_t a, uint32_t size, int reads, struct Route *r)
{
    uint32_t region = a >> 24;
    r->target = 0;
    r->kind = 0;
    r->why = "";
    if (size == 0)
        size = 1;

    if (a < BIOS_SIZE)
    {
        if (!reads)
        {
            r->target = (uint32_t)sShadow + SHADOW_SCRATCH + (a & 0xFFF);
            r->kind = PN_SITE_BIOS_WRITE;
            return ROUTE_OK;
        }
        if (a + size > BIOS_SIZE)
        {
            r->why = "a read crosses from the BIOS region into unmapped memory (CPU open bus)";
            r->kind = PN_SITE_OPEN_BUS;
            return ROUTE_INEXACT;
        }
        RefreshBios(a, size);
        r->target = (uint32_t)sShadow + SHADOW_BIOS + a;
        r->kind = PN_SITE_BIOS_READ;
        return ROUTE_OK;
    }
    switch (region)
    {
    case 0x02:
        if (a < 0x02040000u && a + size > 0x02040000u)
            break;
        return MirrorTarget(0x02000000u, 0x3FFFFu, a, size, r);
    case 0x03:
        if (a < 0x03008000u && a + size > 0x03008000u)
            break;
        return MirrorTarget(0x03000000u, 0x7FFFu, a, size, r);
    case 0x05:
        return MirrorTarget(0x05000000u, 0x3FFu, a, size, r);
    case 0x06:
    {
        uint32_t off = a & 0x1FFFFu;
        uint16_t dispcnt = *(volatile uint16_t *)0x04000000u;
        if (off >= 0x18000u && (off & 0x1C000u) == 0x18000u && (dispcnt & 7) >= 3)
        {
            // mGBA: in bitmap modes 0x06018000-0x0601BFFF reads 0, writes are ignored.
            Fill32(sShadow + SHADOW_ZERO, 0, 0x100);
            r->target = (uint32_t)sShadow + (reads ? SHADOW_ZERO : SHADOW_SCRATCH) + (a & 0x7F);
            r->kind = PN_SITE_MIRROR;
            return ROUTE_OK;
        }
        if (off >= 0x18000u)
            off &= 0x17FFFu;
        if (off + size > 0x18000u)
            break;
        r->target = 0x06000000u + off;
        r->kind = PN_SITE_MIRROR;
        return ROUTE_OK;
    }
    case 0x07:
        return MirrorTarget(0x07000000u, 0x3FFu, a, size, r);
    case 0x0E:
    case 0x0F:
        if (!reads)
        {
            r->why = "a direct write to the flash chip's address space outside its command protocol";
            return ROUTE_INTERNAL;
        }
        r->target = 0x0E000000u + (uint32_t)(gPortFlashBank & 1) * 0x10000u + (a & 0xFFFFu);
        if ((a & 0xFFFFu) + size > 0x10000u)
            break;
        r->kind = PN_SITE_MIRROR;
        return ROUTE_OK;
    }
    if (region >= 0x08 && region <= 0x0D)
    {
        if (!reads)
        {
            r->target = (uint32_t)sShadow + SHADOW_SCRATCH + (a & 0xFFF);
            r->kind = PN_SITE_ROM_WRITE;
            return ROUTE_OK;
        }
        r->why = "a read outside the native image in the cartridge ROM window (the retail ROM's bytes "
                 "there are not reproduced)";
        r->kind = PN_SITE_OPEN_BUS;
        return ROUTE_INEXACT;
    }
    if (region >= 0x02 && region <= 0x07)
    {
        // I/O beyond 0x04000400 (no mirror), or an access crossing a region end.
        if (region == 0x04 || !NativelyMapped(a, 1))
        {
            if (!reads)
            {
                r->target = (uint32_t)sShadow + SHADOW_SCRATCH + (a & 0xFFF);
                r->kind = PN_SITE_UNMAPPED_WRITE;
                return ROUTE_OK;
            }
            if (region == 0x04)
            {
                r->why = "a read of an unused I/O address (CPU open bus)";
                r->kind = PN_SITE_OPEN_BUS;
                return ROUTE_INEXACT;
            }
        }
        r->why = "an access crosses the end of a GBA memory region";
        return ROUTE_INTERNAL;
    }
    // 0x00004000-0x01FFFFFF and 0x10000000 and up: unmapped on the GBA.
    if (!reads)
    {
        r->target = (uint32_t)sShadow + SHADOW_SCRATCH + (a & 0xFFF);
        r->kind = PN_SITE_UNMAPPED_WRITE;
        return ROUTE_OK;
    }
    r->why = "a read of unmapped GBA memory returns the CPU open bus (the prefetched game opcode), "
             "which native code cannot reproduce";
    r->kind = PN_SITE_OPEN_BUS;
    return ROUTE_INEXACT;
}

// Element access through the GBA bus (string instruction emulation).
static int BusRead(uint32_t a, uint32_t size, uint32_t *out, struct Route *r)
{
    uint32_t v = 0, i;
    if (NativelyMapped(a, size) || (a >= 0x08000000u && a + size <= sImageEnd))
    {
        for (i = 0; i < size; i++)
            v |= (uint32_t)*(volatile const uint8_t *)(a + i) << (8 * i);
        *out = v;
        r->kind = 0;
        return ROUTE_OK;
    }
    if (a < BIOS_SIZE && a + size <= BIOS_SIZE)
    {
        uint32_t p = BiosWord();
        for (i = 0; i < size; i++)
            v |= ((p >> (8 * ((a + i) & 3))) & 0xFF) << (8 * i);
        *out = v;
        r->kind = PN_SITE_BIOS_READ;
        return ROUTE_OK;
    }
    {
        int rc = RouteAccess(a, size, 1, r);
        if (rc != ROUTE_OK)
            return rc;
        for (i = 0; i < size; i++)
            v |= (uint32_t)*(volatile const uint8_t *)(r->target + i) << (8 * i);
        *out = v;
        return ROUTE_OK;
    }
}

static int BusWrite(uint32_t a, uint32_t size, uint32_t v, struct Route *r)
{
    uint32_t i, t = a;
    r->kind = 0;
    // Host data of the executable (not GBA memory): written as the instruction would.
    if (!NativelyMapped(a, size) && !(a >= sImageRwStart && a + size <= sImageRwEnd))
    {
        int rc = RouteAccess(a, size, 0, r);
        if (rc != ROUTE_OK)
            return rc;
        t = r->target;
    }
    for (i = 0; i < size; i++)
        *(volatile uint8_t *)(t + i) = (uint8_t)(v >> (8 * i));
    return ROUTE_OK;
}

// ---- explicit open-bus loads (PORTABLE patches) ------------------------------------------
//
// A retail load through a pointer that may lie in unmapped GBA memory, made by
// a known Thumb instruction in the ROM: on the GBA the CPU open bus there is the
// opcode two instructions after it (mGBA GBALoadBad: prefetch[1], doubled in
// Thumb mode for code in the cartridge ROM), selected by the address lanes like
// a BIOS-region load (GBALoad8/16/32). Readable native memory (GBA RAM, the
// executable's data) is read normally, the BIOS region gives the BIOS open
// bus. Used by patches that know the ROM instruction (game/NATIVE_CHANGES.md).
uint32_t Port_LoadWithOpenBus(const void *ptr, uint32_t size, uint32_t thumbOpcodeAtPcPlus4)
{
    uint32_t a = (uint32_t)ptr, v = 0, i, bus;
    struct Route r;
    if (size != 1 && size != 2 && size != 4)
        Port_Terminal(PORT_TERMINAL_INTERNAL, "Port_LoadWithOpenBus: size %u", size);
    if (NativelyMapped(a, size) || (a >= (uint32_t)__executable_start && a + size <= sImageEnd))
    {
        for (i = 0; i < size; i++)
            v |= (uint32_t)*(volatile const uint8_t *)(a + i) << (8 * i);
        return v;
    }
    if (a + size <= BIOS_SIZE)
    {
        BusRead(a, size, &v, &r);
        RecordExplicit((uint32_t)__builtin_return_address(0), PN_SITE_EXPLICIT_BIOS, a, BiosWord(),
                       (uint32_t)__builtin_frame_address(0));
        return v;
    }
    if (RouteAccess(a, size, 1, &r) != ROUTE_INEXACT || (a >> 24) - 0x08u <= 0x05u)
        Port_Terminal(PORT_TERMINAL_INEXACT, "open-bus load at 0x%08x: a GBA memory region that is not modelled "
                      "for this access", a);
    if ((size == 2 && (a & 1)) || (size == 4 && (a & 3)))
        Port_Terminal(PORT_TERMINAL_INEXACT, "misaligned open-bus load at 0x%08x (ARM rotates it)", a);
    bus = (thumbOpcodeAtPcPlus4 & 0xFFFFu) * 0x10001u;
    v = size == 4 ? bus : (bus >> (8 * (a & (size == 2 ? 2u : 3u)))) & (size == 2 ? 0xFFFFu : 0xFFu);
    RecordExplicit((uint32_t)__builtin_return_address(0), PN_SITE_EXPLICIT_OPEN_BUS, a, bus,
                   (uint32_t)__builtin_frame_address(0));
    return v;
}

// ---- ending the environment from a signal handler -------------------------------------

static void EndFromSignal(greg_t *g, uint32_t kind, uint32_t eip, uint32_t addr, const char *why)
{
    char msg[240];
    struct Out o = {msg, 0, sizeof(msg)};
    uint32_t top = ((uint32_t)sRescueStack + sizeof(sRescueStack)) & ~15u;
    PutS(&o, why);
    if (addr || kind == PORT_TERMINAL_INEXACT)
    {
        PutS(&o, " (GBA address ");
        PutHex(&o, addr, 8);
        PutS(&o, ")");
    }
    if (Port_GetTerminal(0) == PORT_TERMINAL_NONE)
    {
        Port_SetTerminalRaw(kind, eip, addr, msg);
        Port_SetTerminalBacktrace((uint32_t)g[G_ESP]);
    }
    RecordSiteBt(eip, kind == PORT_TERMINAL_INEXACT ? PN_SITE_OPEN_BUS : PN_SITE_TERMINAL, addr, 0,
                 (uint32_t)g[G_ESP]);
    // "Call" Port_TerminalThunk on the rescue stack; it leaves the frame
    // through frame.c's __builtin_longjmp target on the game stack.
    *(uint32_t *)(top - 4) = 0; // fake return address
    g[G_ESP] = (greg_t)(top - 4);
    g[G_EBP] = 0;
    g[G_EIP] = (greg_t)(uint32_t)Port_TerminalThunk;
    g[G_EFL] &= ~(greg_t)(EFL_DF | EFL_TF);
}

// ---- trampoline ----------------------------------------------------------------------

static void Put32(uint8_t *p, uint32_t v)
{
    p[0] = (uint8_t)v;
    p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16);
    p[3] = (uint8_t)(v >> 24);
}

// The instruction at `code` with its memory operand replaced by the absolute
// address `target`, followed by a jump to `next`.
static uint32_t BuildTrampoline(const uint8_t *code, const struct X86Insn *d, uint32_t target, uint32_t next)
{
    uint8_t *t = sTramp;
    uint32_t n = 0, i;
    uint32_t opEnd = d->memKind == X86_MEM_MOFFS ? d->dispOff : d->modrmOff;
    for (i = 0; i < opEnd; i++)
        t[n++] = code[i];
    if (d->memKind == X86_MEM_MODRM)
        t[n++] = (uint8_t)((d->modrm & 0x38) | 0x05); // mod 00, r/m 101: [disp32]
    Put32(t + n, target);
    n += 4;
    for (i = 0; i < d->immSize; i++)
        t[n++] = code[d->immOff + i];
    t[n] = 0xE9; // jmp rel32
    Put32(t + n + 1, next - ((uint32_t)t + n + 5));
    __sync_synchronize();
    return (uint32_t)t;
}

// ---- the handlers ------------------------------------------------------------------------

static void Chain(int sig, siginfo_t *si, void *ctx)
{
    struct sigaction *old = sig == SIGSEGV ? &sOldSegv : sig == SIGBUS ? &sOldBus
                          : sig == SIGFPE ? &sOldFpe : &sOldIll;
    if ((old->sa_flags & SA_SIGINFO) && old->sa_sigaction)
    {
        old->sa_sigaction(sig, si, ctx);
        return;
    }
    if (!(old->sa_flags & SA_SIGINFO) && old->sa_handler != SIG_DFL && old->sa_handler != SIG_IGN)
    {
        old->sa_handler(sig);
        return;
    }
    // Default action: restore it; the instruction faults again and the
    // process terminates as it would have without this module.
    {
        struct sigaction dfl;
        uint8_t *p = (uint8_t *)&dfl;
        uint32_t i;
        for (i = 0; i < sizeof(dfl); i++)
            p[i] = 0;
        dfl.sa_handler = SIG_DFL;
        sigaction(sig, &dfl, 0);
    }
}

static void LoadRegs(const greg_t *g, uint32_t regs[8])
{
    uint32_t r;
    for (r = 0; r < 8; r++)
        regs[r] = (uint32_t)g[GREG(r)];
}

// 32-bit C division (div/idiv r/m32 with EDX the extension of EAX).
static int HandleDivide(greg_t *g, const struct X86Insn *d, uint32_t eip)
{
    uint32_t divisor, eax = (uint32_t)g[G_EAX], edx = (uint32_t)g[G_EDX], q, kind;
    struct Route r;
    if (d->map != 0 || d->op != 0xF7 || d->opsize16 || (d->reg != 6 && d->reg != 7))
        return -1;
    if (d->memKind == X86_MEM_NONE)
    {
        divisor = (uint32_t)g[GREG(d->rm)];
    }
    else
    {
        uint32_t regs[8];
        LoadRegs(g, regs);
        if (BusRead(X86EffectiveAddress(d, regs), 4, &divisor, &r) != ROUTE_OK)
            return -1;
    }
    if (d->reg == 6)
    {
        // DIV: faults only for divisor 0 when EDX == 0.
        if (edx != 0 || divisor != 0)
            return -1;
        q = 0;
        kind = PN_SITE_DIV_ZERO;
    }
    else
    {
        if (edx != (uint32_t)((int32_t)eax >> 31))
            return -1;
        if (divisor == 0)
        {
            q = 0;
            kind = PN_SITE_DIV_ZERO;
        }
        else if (divisor == 0xFFFFFFFFu && eax == 0x80000000u)
        {
            q = 0x80000000u;
            kind = PN_SITE_DIV_OVERFLOW;
        }
        else
        {
            return -1;
        }
    }
    // agbcc's __divsi3/__modsi3/__udivsi3/__umodsi3 (lib1thumb.asm): division by
    // zero returns 0 (quotient and remainder); INT_MIN / -1 = INT_MIN, % = 0.
    g[G_EAX] = (greg_t)q;
    g[G_EDX] = 0;
    g[G_EIP] = (greg_t)(eip + d->len);
    RecordSiteBt(eip, kind, 0, eax, 0);
    return 0;
}

// rep movs / stos / lods with an operand in GBA memory that is not natively
// mapped.
static int HandleString(greg_t *g, const struct X86Insn *d, uint32_t eip)
{
    uint32_t size = d->memSize, esi = (uint32_t)g[G_ESI], edi = (uint32_t)g[G_EDI];
    uint32_t ecx = (uint32_t)g[G_ECX], eax = (uint32_t)g[G_EAX];
    uint32_t step = ((uint32_t)g[G_EFL] & EFL_DF) ? (uint32_t)-(int32_t)size : size;
    uint32_t firstKind = 0, firstAddr = 0, firstValue = 0;
    int rep = d->rep != 0;
    struct Route r;

    if (d->strKind != X86_STR_MOVS && d->strKind != X86_STR_STOS && d->strKind != X86_STR_LODS)
    {
        EndFromSignal(g, PORT_TERMINAL_INTERNAL, eip, (uint32_t)g[G_ESI],
                      "a CMPS/SCAS/INS/OUTS string instruction faulted in game code (not emulated)");
        return 0;
    }
    for (;;)
    {
        uint32_t v = 0;
        int rc;
        if (rep && ecx == 0)
            break;
        if (d->strKind != X86_STR_STOS)
        {
            rc = BusRead(esi, size, &v, &r);
            if (rc != ROUTE_OK)
            {
                g[G_ESI] = (greg_t)esi;
                g[G_EDI] = (greg_t)edi;
                g[G_ECX] = (greg_t)ecx;
                EndFromSignal(g, rc == ROUTE_INEXACT ? PORT_TERMINAL_INEXACT : PORT_TERMINAL_INTERNAL, eip,
                              esi, r.why);
                return 0;
            }
            if (r.kind && !firstKind)
            {
                firstKind = r.kind;
                firstAddr = esi;
                firstValue = BiosWord();
            }
        }
        else
        {
            v = eax;
        }
        if (d->strKind != X86_STR_LODS)
        {
            rc = BusWrite(edi, size, v, &r);
            if (rc != ROUTE_OK)
            {
                g[G_ESI] = (greg_t)esi;
                g[G_EDI] = (greg_t)edi;
                g[G_ECX] = (greg_t)ecx;
                EndFromSignal(g, PORT_TERMINAL_INTERNAL, eip, edi, r.why);
                return 0;
            }
            if (r.kind && !firstKind)
            {
                firstKind = r.kind;
                firstAddr = edi;
            }
            edi += step;
        }
        else
        {
            if (size == 1)
                eax = (eax & ~0xFFu) | (v & 0xFF);
            else if (size == 2)
                eax = (eax & ~0xFFFFu) | (v & 0xFFFF);
            else
                eax = v;
        }
        if (d->strKind != X86_STR_STOS)
            esi += step;
        if (!rep)
            break;
        ecx--;
    }
    g[G_ESI] = (greg_t)esi;
    g[G_EDI] = (greg_t)edi;
    g[G_ECX] = (greg_t)ecx;
    g[G_EAX] = (greg_t)eax;
    g[G_EIP] = (greg_t)(eip + d->len);
    RecordSiteBt(eip, firstKind ? firstKind : PN_SITE_MIRROR, firstAddr, firstValue, 0);
    return 0;
}

static void HandleMemoryFault(greg_t *g, uint32_t eip, uint32_t cr2)
{
    const uint8_t *code = (const uint8_t *)eip;
    struct X86Insn d;
    struct Route r;
    uint32_t regs[8], ea, size, esp = (uint32_t)g[G_ESP];
    int rc, reads;

    if (InTrampoline(eip))
    {
        EndFromSignal(g, PORT_TERMINAL_INTERNAL, eip, cr2, "a redirected access faulted again");
        return;
    }
    if (!InText(eip))
    {
        // An instruction fetch: the game called/jumped to a non-code address.
        EndFromSignal(g, PORT_TERMINAL_CRASHED, eip, eip,
                      eip < BIOS_SIZE ? "the game jumped into the BIOS region (on the GBA: BIOS vectors, "
                                        "e.g. the reset vector restarts the ROM without a RAM reset)"
                                      : "the game jumped to an address that holds no code");
        return;
    }
    // Game stack overflow: the stack is at the bottom of .pnstate and nothing
    // is mapped below it.
    if (cr2 < StackLo() && cr2 >= StackLo() - 0x10000 && esp < StackLo() + 0x1000)
    {
        EndFromSignal(g, PORT_TERMINAL_INTERNAL, eip, 0,
                      "game stack overflow (raise PORT_GAME_STACK_SIZE in platform/port_fault.h)");
        return;
    }
    if (X86Decode(code, &d) != 0)
    {
        EndFromSignal(g, PORT_TERMINAL_INTERNAL, eip, cr2, "a faulting game instruction could not be decoded");
        return;
    }
    if (d.memKind == X86_MEM_STRING)
    {
        HandleString(g, &d, eip);
        return;
    }
    if (d.memKind == X86_MEM_NONE || (d.access & X86_ACC_NONE))
    {
        EndFromSignal(g, PORT_TERMINAL_INTERNAL, eip, cr2,
                      "a game instruction without a data memory operand faulted (stack?)");
        return;
    }
    if (d.seg == 0x64 || d.seg == 0x65)
    {
        EndFromSignal(g, PORT_TERMINAL_INTERNAL, eip, cr2, "an FS/GS-relative access faulted in game code");
        return;
    }
    LoadRegs(g, regs);
    ea = X86EffectiveAddress(&d, regs);
    size = d.memSize ? d.memSize : 4;
    // The fault must be the memory operand's (not a push/call on the stack).
    if (cr2 - ea >= (d.memSize > 16 ? d.memSize : 16))
    {
        EndFromSignal(g, PORT_TERMINAL_INTERNAL, eip, cr2,
                      "a game instruction faulted outside its memory operand");
        return;
    }
    if (d.access & X86_ACC_CONTROL)
    {
        EndFromSignal(g, PORT_TERMINAL_CRASHED, eip, ea,
                      "the game called/jumped through a code pointer read from unmapped memory (the GBA "
                      "branches to the BIOS open-bus value)");
        return;
    }
    reads = (d.access & X86_ACC_READ) != 0;
    rc = RouteAccess(ea, size, reads, &r);
    if (rc != ROUTE_OK)
    {
        EndFromSignal(g, rc == ROUTE_INEXACT ? PORT_TERMINAL_INEXACT : PORT_TERMINAL_INTERNAL, eip, ea, r.why);
        return;
    }
    if (r.kind == PN_SITE_BIOS_READ && d.memSize == 2 && (ea & 1))
    {
        // ARM LDRH at an odd address rotates the halfword; x86 reads bytes a, a+1.
        EndFromSignal(g, PORT_TERMINAL_INEXACT, eip, ea,
                      "a halfword read at an odd BIOS-region address (ARM rotates it)");
        return;
    }
    RecordSiteBt(eip, r.kind, ea, r.kind == PN_SITE_BIOS_READ ? BiosWord() : 0, 0);
    g[G_EIP] = (greg_t)BuildTrampoline(code, &d, r.target, eip + d.len);
}

static void FaultHandler(int sig, siginfo_t *si, void *ctxv)
{
    ucontext_t *uc = (ucontext_t *)ctxv;
    greg_t *g = uc->uc_mcontext.gregs;
    uint32_t eip = (uint32_t)g[G_EIP];
    struct X86Insn d;

    if (!IsGameContext((uint32_t)g[G_ESP]))
    {
        Chain(sig, si, ctxv);
        return;
    }
    switch (sig)
    {
    case SIGSEGV:
    case SIGBUS:
        if (g[G_TRAPNO] != 14 && InText(eip))
        {
            // Not a page fault (#GP: privileged or misaligned SSE access).
            EndFromSignal(g, PORT_TERMINAL_CRASHED, eip, (uint32_t)si->si_addr,
                          "general protection fault in game code");
            return;
        }
        HandleMemoryFault(g, eip, (uint32_t)uc->uc_mcontext.cr2);
        return;
    case SIGFPE:
        if (InText(eip) && X86Decode((const uint8_t *)eip, &d) == 0 && HandleDivide(g, &d, eip) == 0)
            return;
        EndFromSignal(g, PORT_TERMINAL_INTERNAL, eip, 0,
                      "arithmetic fault in game code that is not a 32-bit C division (not modelled)");
        return;
    case SIGILL:
        EndFromSignal(g, PORT_TERMINAL_CRASHED, eip, 0,
                      "illegal instruction in game code (a GCC trap for undefined behaviour, or a jump "
                      "into data)");
        return;
    }
    Chain(sig, si, ctxv);
}

// ---- hang watchdog --------------------------------------------------------------------------

static void WatchdogHandler(int sig, siginfo_t *si, void *ctxv)
{
    ucontext_t *uc = (ucontext_t *)ctxv;
    greg_t *g = uc->uc_mcontext.gregs;
    uint32_t eip = (uint32_t)g[G_EIP], esp = (uint32_t)g[G_ESP];
    (void)sig;
    (void)si;
    if (!IsGameContext(esp) || esp < StackLo() || !(InText(eip) || InTrampoline(eip)))
    {
        // Host code, or game code inside the C library (fatal path): next tick.
        if (!gPortOnGameStack)
            sWatchArmed = 0;
        return;
    }
    if (sWatchArmed && sWatchSeq == gPortGameRunSeq)
    {
        sWatchArmed = 0;
        EndFromSignal(g, PORT_TERMINAL_HUNG, eip, 0,
                      "hang watchdog: one frame body ran for more than the watchdog period of CPU time "
                      "(a native loop that cannot end; see the site for where it spun)");
        return;
    }
    sWatchArmed = 1;
    sWatchSeq = gPortGameRunSeq;
}

void Port_SetHangWatchdog(uint32_t cpu_ms)
{
    sWatchdogCfg = cpu_ms ? cpu_ms : 0xFFFFFFFFu;
    sPreparedValid = 0; // re-arm on the next frame
}

static void ArmWatchdog(void)
{
    struct sigevent sev;
    struct itimerspec its;
    uint32_t ms = sWatchdogCfg == 0 ? WATCHDOG_DEFAULT_MS : sWatchdogCfg;
    uint8_t *p = (uint8_t *)&sev;
    uint32_t i;

    if (sWatchdogTimerValid)
    {
        timer_delete(sWatchdogTimer);
        sWatchdogTimerValid = 0;
    }
    sWatchArmed = 0;
    if (ms == 0xFFFFFFFFu)
        return;
    for (i = 0; i < sizeof(sev); i++)
        p[i] = 0;
    sev.sigev_notify = SIGEV_THREAD_ID;
    sev.sigev_signo = SIGVTALRM;
    sev._sigev_un._tid = (int)syscall(SYS_gettid);
    if (timer_create(CLOCK_THREAD_CPUTIME_ID, &sev, &sWatchdogTimer) != 0)
        return;
    sWatchdogTimerValid = 1;
    its.it_interval.tv_sec = ms / 1000;
    its.it_interval.tv_nsec = (long)(ms % 1000) * 1000000L;
    its.it_value = its.it_interval;
    timer_settime(sWatchdogTimer, 0, &its, 0);
}

// ---- installation ---------------------------------------------------------------------------

static void Reserve(uint32_t lo, uint32_t hi)
{
    void *p;
    if (hi <= lo)
        return;
    p = mmap((void *)lo, hi - lo, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED_NOREPLACE,
             -1, 0);
    if (p != MAP_FAILED && p != (void *)lo)
    {
        munmap(p, hi - lo); // kernel without MAP_FIXED_NOREPLACE: it was only a hint
        p = MAP_FAILED;
    }
    if (p == MAP_FAILED && hi - lo > 0x10000)
    {
        // Something is mapped inside: reserve the free parts.
        uint32_t mid = lo + (((hi - lo) / 2) & ~0xFFFFu);
        Reserve(lo, mid);
        Reserve(mid, hi);
    }
}

static uint32_t ImageEnd(void)
{
    const Elf32_Ehdr *eh = (const Elf32_Ehdr *)__executable_start;
    const Elf32_Phdr *ph = (const Elf32_Phdr *)(__executable_start + eh->e_phoff);
    uint32_t i, end = (uint32_t)etext;
    for (i = 0; i < eh->e_phnum; i++)
    {
        if (ph[i].p_type != PT_LOAD || ph[i].p_vaddr < 0x08000000u || ph[i].p_vaddr >= 0x0E000000u)
            continue;
        if (ph[i].p_vaddr + ph[i].p_memsz > end)
            end = ph[i].p_vaddr + ph[i].p_memsz;
        if ((ph[i].p_flags & PF_W) && !sImageRwStart)
        {
            sImageRwStart = ph[i].p_vaddr;
            sImageRwEnd = ph[i].p_vaddr + ph[i].p_memsz;
        }
    }
    return end;
}

// A forked child keeps the signal handlers and the alternate stack of the
// forking thread, but not its CPU-time timer: arm the watchdog again.
static void AfterForkInChild(void)
{
    sWatchdogTimerValid = 0;
    sPreparedValid = 0;
}

static void InstallProcessWide(void)
{
    struct sigaction sa;
    uint8_t *p = (uint8_t *)&sa;
    uint32_t i, k;
    static const uint32_t kPatterns[] = {0xE3A02004u, 0xE25EF004u, 0xE55EC002u};

    sInstalled = 1;
    sShadow = mmap(0, SHADOW_SIZE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    sTramp = mmap(0, 4096, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (sShadow == MAP_FAILED || sTramp == MAP_FAILED)
    {
        sShadow = 0;
        sTramp = 0;
        sInstallFailed = 1;
        WriteErr("fireredsim: cannot map the fault-emulation pages; retail NULL accesses will crash\n");
        return;
    }
    sImageEnd = ImageEnd();

    // Keep host mappings out of the GBA address space below 0x10000000 and out
    // of the windows that the BIOS open-bus words point to (stale pointers read
    // from the BIOS region): accesses there must fault (and be handled), never
    // reach host memory.
    Reserve(0x00010000u, 0x02000000u);
    Reserve(0x02040000u, 0x03000000u);
    Reserve(0x03008000u, 0x04000000u);
    Reserve(0x04001000u, 0x05000000u);
    Reserve(0x05001000u, 0x06000000u);
    Reserve(0x06018000u, 0x07000000u);
    Reserve(0x07001000u, 0x08000000u);
    Reserve((sImageEnd + 0xFFFFu) & ~0xFFFFu, 0x0E000000u);
    Reserve(0x0E020000u, 0x10000000u - 0x10000u);
    for (k = 0; k < sizeof(kPatterns) / sizeof(kPatterns[0]); k++)
        Reserve((kPatterns[k] & ~0xFFFFu) - 0x10000u, (kPatterns[k] & ~0xFFFFu) + 0x20000u);

    for (i = 0; i < sizeof(sa); i++)
        p[i] = 0;
    sa.sa_sigaction = FaultHandler;
    sa.sa_flags = SA_SIGINFO | SA_ONSTACK;
    sigemptyset(&sa.sa_mask);
    sigaddset(&sa.sa_mask, SIGVTALRM);
    sigaction(SIGSEGV, &sa, &sOldSegv);
    sigaction(SIGBUS, &sa, &sOldBus);
    sigaction(SIGFPE, &sa, &sOldFpe);
    sigaction(SIGILL, &sa, &sOldIll);

    sa.sa_sigaction = WatchdogHandler;
    sa.sa_flags = SA_SIGINFO | SA_ONSTACK | SA_RESTART;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGVTALRM, &sa, &sOldWatch);
    pthread_atfork(0, 0, AfterForkInChild);
}

static void PrepareSlow(void)
{
    stack_t ss, old;
    if (!sInstalled)
        InstallProcessWide();
    if (sInstallFailed)
    {
        sPreparedThread = pthread_self();
        sPreparedValid = 1;
        return;
    }
    // Alternate signal stack for this thread (the fault must not push a
    // signal frame onto the game stack, which is environment state).
    if (sigaltstack(0, &old) != 0 || (old.ss_flags & SS_DISABLE) || old.ss_size < 0x10000)
    {
        void *mem = mmap(0, 0x10000, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (mem != MAP_FAILED)
        {
            ss.ss_sp = mem;
            ss.ss_size = 0x10000;
            ss.ss_flags = 0;
            sigaltstack(&ss, 0);
        }
    }
    ArmWatchdog();
    sPreparedThread = pthread_self();
    sPreparedValid = 1;
}

void Port_FaultPrepareThread(void)
{
    if (sPreparedValid && pthread_equal(pthread_self(), sPreparedThread))
        return;
    PrepareSlow();
}
