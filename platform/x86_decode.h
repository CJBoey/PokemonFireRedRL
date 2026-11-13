// fireredsim platform layer: a small IA-32 instruction decoder.
//
// Used by the memory fault handler (platform/memfault.c) to find the memory
// operand of a faulting game-code instruction, so that the access can be
// redirected to the GBA's value for that address (BIOS open bus, mirrors) and
// the instruction re-executed unchanged otherwise. It decodes the 32-bit
// protected-mode encoding GCC emits for i386/SSE2 code: prefixes, the one-byte,
// 0F, 0F38 and 0F3A opcode maps, ModRM/SIB/displacement, immediates, plus a
// table of the memory access each opcode makes (size and read/write). It does
// not need to know what an instruction computes.
//
// Validated against objdump over every instruction of the executable
// (tests/fuzz: test_x86_decode).
//
// Plain 32-bit fields only: the struct has the same layout in platform code
// (compiled with the force-included ARM ABI macro) and in host C++ code.
#ifndef FIREREDSIM_X86_DECODE_H
#define FIREREDSIM_X86_DECODE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Kind of memory operand.
enum {
    X86_MEM_NONE = 0,   // no data memory operand (or ModRM with mod == 3)
    X86_MEM_MODRM = 1,  // ModRM/SIB memory operand
    X86_MEM_MOFFS = 2,  // A0-A3: absolute 32-bit address (no ModRM)
    X86_MEM_STRING = 3, // MOVS/CMPS/STOS/LODS/SCAS/INS/OUTS: implicit [ESI] / [EDI]
};

// Access flags.
enum {
    X86_ACC_READ = 1,
    X86_ACC_WRITE = 2,
    X86_ACC_CONTROL = 4, // CALL/JMP through memory (the value read is a code address)
    X86_ACC_NONE = 8,    // LEA, NOP r/m, PREFETCH: a ModRM memory form without an access
};

// String instruction kinds (memKind == X86_MEM_STRING).
enum {
    X86_STR_MOVS = 1,
    X86_STR_CMPS = 2,
    X86_STR_STOS = 3,
    X86_STR_LODS = 4,
    X86_STR_SCAS = 5,
    X86_STR_INS = 6,
    X86_STR_OUTS = 7,
};

// Register numbers as encoded (EAX=0 ECX EDX EBX ESP EBP ESI EDI).
#define X86_REG_NONE 0xFFu

struct X86Insn
{
    uint32_t len;        // instruction length in bytes
    uint32_t nprefix;    // prefix bytes before the opcode
    uint32_t opsize16;   // 0x66 seen
    uint32_t addr16;     // 0x67 seen (16-bit addressing: not supported by the handler)
    uint32_t rep;        // 0xF2 / 0xF3 or 0
    uint32_t lock;       // 0xF0 seen
    uint32_t seg;        // segment override prefix byte or 0
    uint32_t map;        // 0: one-byte, 1: 0F xx, 2: 0F 38 xx, 3: 0F 3A xx
    uint32_t op;         // last opcode byte
    uint32_t hasModrm;
    uint32_t modrmOff;   // offset of the ModRM byte
    uint32_t modrm;
    uint32_t mod, reg, rm;
    uint32_t hasSib;
    uint32_t sib;
    uint32_t base;       // base register or X86_REG_NONE
    uint32_t index;      // index register or X86_REG_NONE
    uint32_t scale;      // 0..3 (index << scale)
    uint32_t dispOff;    // offset of the displacement (ModRM forms) or moffs
    uint32_t dispSize;   // 0, 1 or 4
    int32_t disp;
    uint32_t immOff;     // offset of the immediate bytes
    uint32_t immSize;
    uint32_t memKind;    // X86_MEM_*
    uint32_t memSize;    // bytes accessed through the memory operand; 0 = unknown (<= 16)
    uint32_t access;     // X86_ACC_* for the memory operand
    uint32_t strKind;    // X86_STR_* for string instructions
};

// Decodes the instruction at `code` (at most 15 bytes are read). Returns 0 on
// success, -1 for an invalid or unsupported encoding.
int X86Decode(const uint8_t *code, struct X86Insn *out);

// Effective address of a ModRM or moffs memory operand with the given register
// values (indexed by register number).
uint32_t X86EffectiveAddress(const struct X86Insn *insn, const uint32_t regs[8]);

#ifdef __cplusplus
}
#endif

#endif // FIREREDSIM_X86_DECODE_H
