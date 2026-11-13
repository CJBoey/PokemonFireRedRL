// fireredsim platform layer: a small IA-32 instruction decoder (see x86_decode.h).
//
// Reference: Intel SDM vol. 2, appendix A (opcode maps). 32-bit code segment,
// 32-bit default operand and address size.
#include "x86_decode.h"

// ---- opcode tables -------------------------------------------------------------
// One bit per opcode: bit (op & 31) of word [op >> 5].

#define B(op) (1u << ((op) & 31))

// One-byte opcodes with a ModRM byte.
static const uint32_t kModrm1[8] = {
    // 00-1F: ALU r/m forms 00-03, 08-0B, 10-13, 18-1B
    B(0x00) | B(0x01) | B(0x02) | B(0x03) | B(0x08) | B(0x09) | B(0x0A) | B(0x0B) |
    B(0x10) | B(0x11) | B(0x12) | B(0x13) | B(0x18) | B(0x19) | B(0x1A) | B(0x1B),
    // 20-3F
    B(0x20) | B(0x21) | B(0x22) | B(0x23) | B(0x28) | B(0x29) | B(0x2A) | B(0x2B) |
    B(0x30) | B(0x31) | B(0x32) | B(0x33) | B(0x38) | B(0x39) | B(0x3A) | B(0x3B),
    // 40-5F: none
    0,
    // 60-7F: BOUND, ARPL, IMUL Gv,Ev,Iz, IMUL Gv,Ev,Ib
    B(0x62) | B(0x63) | B(0x69) | B(0x6B),
    // 80-9F: 80-8F
    0xFFFFu,
    // A0-BF: none
    0,
    // C0-DF: C0 C1 C4 C5 C6 C7, D0-D3, D8-DF
    B(0xC0) | B(0xC1) | B(0xC4) | B(0xC5) | B(0xC6) | B(0xC7) |
    B(0xD0) | B(0xD1) | B(0xD2) | B(0xD3) | B(0xD8) | B(0xD9) | B(0xDA) | B(0xDB) |
    B(0xDC) | B(0xDD) | B(0xDE) | B(0xDF),
    // E0-FF: F6 F7 FE FF
    B(0xF6) | B(0xF7) | B(0xFE) | B(0xFF),
};

// Two-byte opcodes (0F xx) WITHOUT a ModRM byte.
static const uint32_t kNoModrm2[8] = {
    // 00-1F: 04-0C (except 0D), 0E
    B(0x04) | B(0x05) | B(0x06) | B(0x07) | B(0x08) | B(0x09) | B(0x0A) | B(0x0B) | B(0x0C) | B(0x0E),
    // 20-3F: 24-27, 30-37, 39, 3B-3F (38/3A are escapes, handled before)
    B(0x24) | B(0x25) | B(0x26) | B(0x27) | B(0x30) | B(0x31) | B(0x32) | B(0x33) |
    B(0x34) | B(0x35) | B(0x36) | B(0x37) | B(0x39) | B(0x3B) | B(0x3C) | B(0x3D) | B(0x3E) | B(0x3F),
    // 40-5F: none
    0,
    // 60-7F: 77 (EMMS)
    B(0x77),
    // 80-9F: 80-8F (Jcc rel32)
    0xFFFFu,
    // A0-BF: A0 A1 A2 A6 A7 A8 A9 AA
    B(0xA0) | B(0xA1) | B(0xA2) | B(0xA6) | B(0xA7) | B(0xA8) | B(0xA9) | B(0xAA),
    // C0-DF: C8-CF (BSWAP)
    B(0xC8) | B(0xC9) | B(0xCA) | B(0xCB) | B(0xCC) | B(0xCD) | B(0xCE) | B(0xCF),
    // E0-FF: none
    0,
};

// Two-byte opcodes (0F xx) with an 8-bit immediate after the ModRM operand.
static const uint32_t kImm8_2[8] = {
    B(0x0F), 0, 0,
    B(0x70) | B(0x71) | B(0x72) | B(0x73),
    0,
    B(0xA4) | B(0xAC) | B(0xBA),
    B(0xC2) | B(0xC4) | B(0xC5) | B(0xC6),
    0,
};

static inline int TestBit(const uint32_t *table, uint32_t op)
{
    return (table[op >> 5] >> (op & 31)) & 1;
}

// ---- memory access model ---------------------------------------------------------

static uint32_t OpSize(const struct X86Insn *d)
{
    return d->opsize16 ? 2 : 4;
}

// Size by SSE mandatory prefix: none / F3 / 66 / F2.
static uint32_t SseSize(const struct X86Insn *d, uint32_t none, uint32_t f3, uint32_t p66, uint32_t f2)
{
    if (d->rep == 0xF3)
        return f3;
    if (d->rep == 0xF2)
        return f2;
    if (d->opsize16)
        return p66;
    return none;
}

static void SetAccess(struct X86Insn *d, uint32_t size, uint32_t access)
{
    d->memSize = size;
    d->access = access;
}

static void MemoryModelOneByte(struct X86Insn *d)
{
    uint32_t op = d->op, reg = d->reg;
    const uint32_t R = X86_ACC_READ, W = X86_ACC_WRITE, RW = X86_ACC_READ | X86_ACC_WRITE;

    if (op < 0x40 && (op & 7) < 4)
    {
        // ALU: op&7 = 0 Eb,Gb  1 Ev,Gv  2 Gb,Eb  3 Gv,Ev; 38-3B = CMP (read only)
        uint32_t size = (op & 1) ? OpSize(d) : 1;
        if ((op & 7) >= 2 || (op & 0x38) == 0x38)
            SetAccess(d, size, R);
        else
            SetAccess(d, size, RW);
        return;
    }
    switch (op)
    {
    case 0x62: SetAccess(d, 2 * OpSize(d), R); return;   // BOUND
    case 0x63: SetAccess(d, 2, RW); return;               // ARPL
    case 0x69: case 0x6B: SetAccess(d, OpSize(d), R); return; // IMUL
    case 0x80: case 0x82: SetAccess(d, 1, reg == 7 ? R : RW); return;
    case 0x81: case 0x83: SetAccess(d, OpSize(d), reg == 7 ? R : RW); return;
    case 0x84: SetAccess(d, 1, R); return;                // TEST
    case 0x85: SetAccess(d, OpSize(d), R); return;
    case 0x86: SetAccess(d, 1, RW); return;               // XCHG
    case 0x87: SetAccess(d, OpSize(d), RW); return;
    case 0x88: SetAccess(d, 1, W); return;                // MOV Eb,Gb
    case 0x89: SetAccess(d, OpSize(d), W); return;
    case 0x8A: SetAccess(d, 1, R); return;
    case 0x8B: SetAccess(d, OpSize(d), R); return;
    case 0x8C: SetAccess(d, 2, W); return;                // MOV Ew,Sreg
    case 0x8D: SetAccess(d, 0, X86_ACC_NONE); return;     // LEA
    case 0x8E: SetAccess(d, 2, R); return;                // MOV Sreg,Ew
    case 0x8F: SetAccess(d, OpSize(d), W); return;        // POP Ev
    case 0xA0: SetAccess(d, 1, R); return;                // MOV AL,moffs
    case 0xA1: SetAccess(d, OpSize(d), R); return;
    case 0xA2: SetAccess(d, 1, W); return;
    case 0xA3: SetAccess(d, OpSize(d), W); return;
    case 0xC0: case 0xD0: case 0xD2: SetAccess(d, 1, RW); return; // shifts
    case 0xC1: case 0xD1: case 0xD3: SetAccess(d, OpSize(d), RW); return;
    case 0xC4: case 0xC5: SetAccess(d, 2 + OpSize(d), R); return; // LES/LDS
    case 0xC6: SetAccess(d, 1, W); return;                // MOV Eb,Ib
    case 0xC7: SetAccess(d, OpSize(d), W); return;        // MOV Ev,Iz
    case 0xD8: SetAccess(d, 4, R); return;                // x87 m32fp arithmetic
    case 0xD9:
        switch (reg)
        {
        case 0: SetAccess(d, 4, R); return;               // FLD m32
        case 2: case 3: SetAccess(d, 4, W); return;       // FST/FSTP m32
        case 4: SetAccess(d, 28, R); return;              // FLDENV
        case 5: SetAccess(d, 2, R); return;               // FLDCW
        case 6: SetAccess(d, 28, W); return;              // FNSTENV
        case 7: SetAccess(d, 2, W); return;               // FNSTCW
        }
        break;
    case 0xDA: SetAccess(d, 4, R); return;                // x87 m32int arithmetic
    case 0xDB:
        switch (reg)
        {
        case 0: SetAccess(d, 4, R); return;               // FILD m32
        case 1: case 2: case 3: SetAccess(d, 4, W); return; // FISTTP/FIST/FISTP m32
        case 5: SetAccess(d, 10, R); return;              // FLD m80
        case 7: SetAccess(d, 10, W); return;              // FSTP m80
        }
        break;
    case 0xDC: SetAccess(d, 8, R); return;                // x87 m64fp arithmetic
    case 0xDD:
        switch (reg)
        {
        case 0: SetAccess(d, 8, R); return;               // FLD m64
        case 1: case 2: case 3: SetAccess(d, 8, W); return; // FISTTP/FST/FSTP m64
        case 4: SetAccess(d, 108, R); return;             // FRSTOR
        case 6: SetAccess(d, 108, W); return;             // FNSAVE
        case 7: SetAccess(d, 2, W); return;               // FNSTSW m16
        }
        break;
    case 0xDE: SetAccess(d, 2, R); return;                // x87 m16int arithmetic
    case 0xDF:
        switch (reg)
        {
        case 0: SetAccess(d, 2, R); return;               // FILD m16
        case 1: case 2: case 3: SetAccess(d, 2, W); return;
        case 4: SetAccess(d, 10, R); return;              // FBLD
        case 5: SetAccess(d, 8, R); return;               // FILD m64
        case 6: SetAccess(d, 10, W); return;              // FBSTP
        case 7: SetAccess(d, 8, W); return;               // FISTP m64
        }
        break;
    case 0xF6: case 0xF7:
    {
        uint32_t size = op == 0xF6 ? 1 : OpSize(d);
        // 0,1 TEST; 2 NOT; 3 NEG; 4 MUL; 5 IMUL; 6 DIV; 7 IDIV
        SetAccess(d, size, (reg == 2 || reg == 3) ? RW : R);
        return;
    }
    case 0xFE: SetAccess(d, 1, RW); return;               // INC/DEC Eb
    case 0xFF:
        switch (reg)
        {
        case 0: case 1: SetAccess(d, OpSize(d), RW); return;          // INC/DEC
        case 2: case 4: SetAccess(d, 4, R | X86_ACC_CONTROL); return; // CALL/JMP near
        case 3: case 5: SetAccess(d, 6, R | X86_ACC_CONTROL); return; // CALL/JMP far
        case 6: SetAccess(d, OpSize(d), R); return;                   // PUSH
        }
        break;
    }
    SetAccess(d, 0, R); // conservative default: reads, size unknown
}

static void MemoryModel0F(struct X86Insn *d)
{
    uint32_t op = d->op, reg = d->reg;
    const uint32_t R = X86_ACC_READ, W = X86_ACC_WRITE, RW = X86_ACC_READ | X86_ACC_WRITE;

    switch (op)
    {
    case 0x10: SetAccess(d, SseSize(d, 16, 4, 16, 8), R); return;  // MOVUPS/MOVSS/MOVUPD/MOVSD load
    case 0x11: SetAccess(d, SseSize(d, 16, 4, 16, 8), W); return;  // store
    case 0x12: SetAccess(d, SseSize(d, 8, 16, 8, 8), R); return;   // MOVLPS/MOVSLDUP/MOVLPD/MOVDDUP
    case 0x13: SetAccess(d, 8, W); return;                         // MOVLPS/MOVLPD store
    case 0x14: case 0x15: SetAccess(d, 16, R); return;             // UNPCKLPS/UNPCKHPS
    case 0x16: SetAccess(d, SseSize(d, 8, 16, 8, 8), R); return;   // MOVHPS/MOVSHDUP/MOVHPD
    case 0x17: SetAccess(d, 8, W); return;
    case 0x18: case 0x19: case 0x1A: case 0x1B: case 0x1C: case 0x1D: case 0x1E: case 0x1F:
        SetAccess(d, 0, X86_ACC_NONE); return;                     // PREFETCH, NOP r/m
    case 0x28: SetAccess(d, 16, R); return;                        // MOVAPS/MOVAPD load
    case 0x29: SetAccess(d, 16, W); return;
    case 0x2A: SetAccess(d, SseSize(d, 8, 4, 8, 4), R); return;    // CVTPI2PS/CVTSI2SS/CVTPI2PD/CVTSI2SD
    case 0x2B: SetAccess(d, 16, W); return;                        // MOVNTPS/MOVNTPD
    case 0x2C: case 0x2D: SetAccess(d, SseSize(d, 8, 4, 16, 8), R); return; // CVT(T)PS2PI/SS2SI/PD2PI/SD2SI
    case 0x2E: case 0x2F: SetAccess(d, d->opsize16 ? 8 : 4, R); return;     // (U)COMISS/(U)COMISD
    case 0x5A: SetAccess(d, SseSize(d, 8, 4, 16, 8), R); return;   // CVTPS2PD/CVTSS2SD/CVTPD2PS/CVTSD2SS
    case 0x6E: SetAccess(d, 4, R); return;                         // MOVD mm/xmm, r/m32
    case 0x6F: SetAccess(d, SseSize(d, 8, 16, 16, 8), R); return;  // MOVQ/MOVDQU/MOVDQA
    case 0x7E:
        if (d->rep == 0xF3)
            SetAccess(d, 8, R);                                    // MOVQ xmm, xmm/m64
        else
            SetAccess(d, 4, W);                                    // MOVD r/m32, mm/xmm
        return;
    case 0x7F: SetAccess(d, SseSize(d, 8, 16, 16, 8), W); return;  // MOVQ/MOVDQU/MOVDQA store
    case 0xA3: SetAccess(d, OpSize(d), R); return;                 // BT
    case 0xAB: case 0xB3: case 0xBB: SetAccess(d, OpSize(d), RW); return; // BTS/BTR/BTC
    case 0xA4: case 0xA5: case 0xAC: case 0xAD: SetAccess(d, OpSize(d), RW); return; // SHLD/SHRD
    case 0xAE:
        switch (reg)
        {
        case 0: SetAccess(d, 512, W); return;                      // FXSAVE
        case 1: SetAccess(d, 512, R); return;                      // FXRSTOR
        case 2: SetAccess(d, 4, R); return;                        // LDMXCSR
        case 3: SetAccess(d, 4, W); return;                        // STMXCSR
        case 7: SetAccess(d, 0, X86_ACC_NONE); return;             // CLFLUSH
        }
        break;
    case 0xAF: SetAccess(d, OpSize(d), R); return;                 // IMUL
    case 0xB0: SetAccess(d, 1, RW); return;                        // CMPXCHG
    case 0xB1: SetAccess(d, OpSize(d), RW); return;
    case 0xB6: case 0xBE: SetAccess(d, 1, R); return;              // MOVZX/MOVSX Gv,Eb
    case 0xB7: case 0xBF: SetAccess(d, 2, R); return;              // MOVZX/MOVSX Gv,Ew
    case 0xB8: case 0xBC: case 0xBD: SetAccess(d, OpSize(d), R); return; // POPCNT/BSF/BSR
    case 0xBA: SetAccess(d, OpSize(d), reg == 4 ? R : RW); return; // BT/BTS/BTR/BTC Ev,Ib
    case 0xC0: SetAccess(d, 1, RW); return;                        // XADD
    case 0xC1: SetAccess(d, OpSize(d), RW); return;
    case 0xC2: SetAccess(d, SseSize(d, 16, 4, 16, 8), R); return;  // CMPPS/CMPSS/CMPPD/CMPSD
    case 0xC3: SetAccess(d, 4, W); return;                         // MOVNTI
    case 0xC4: SetAccess(d, 2, R); return;                         // PINSRW
    case 0xC6: SetAccess(d, 16, R); return;                        // SHUFPS/SHUFPD
    case 0xC7: SetAccess(d, 8, RW); return;                        // CMPXCHG8B
    case 0xD6: SetAccess(d, 8, W); return;                         // MOVQ xmm/m64, xmm
    case 0xE7: SetAccess(d, d->opsize16 ? 16 : 8, W); return;      // MOVNTQ/MOVNTDQ
    case 0xF0: SetAccess(d, 16, R); return;                        // LDDQU
    }
    if (op >= 0x40 && op <= 0x4F)
    {
        SetAccess(d, OpSize(d), R); // CMOVcc
        return;
    }
    if (op >= 0x90 && op <= 0x9F)
    {
        SetAccess(d, 1, W); // SETcc
        return;
    }
    if (op >= 0x51 && op <= 0x5F)
    {
        SetAccess(d, SseSize(d, 16, 4, 16, 8), R); // SSE arithmetic
        return;
    }
    if ((op >= 0x60 && op <= 0x6D) || (op >= 0x70 && op <= 0x76) || op >= 0xD0)
    {
        SetAccess(d, d->opsize16 ? 16 : 8, R); // MMX / SSE2 packed integer
        return;
    }
    SetAccess(d, 0, R);
}

// ---- decoder -------------------------------------------------------------------------

int X86Decode(const uint8_t *code, struct X86Insn *d)
{
    uint32_t i = 0, op, immSize = 0, hasModrm = 0;
    uint32_t k;
    uint8_t *raw = (uint8_t *)d;

    for (k = 0; k < sizeof(*d); k++)
        raw[k] = 0;
    d->base = X86_REG_NONE;
    d->index = X86_REG_NONE;

    // Prefixes (at most 14; an instruction is at most 15 bytes).
    for (;;)
    {
        uint8_t b = code[i];
        if (b == 0x66)
            d->opsize16 = 1;
        else if (b == 0x67)
            d->addr16 = 1;
        else if (b == 0xF2 || b == 0xF3)
            d->rep = b;
        else if (b == 0xF0)
            d->lock = 1;
        else if (b == 0x26 || b == 0x2E || b == 0x36 || b == 0x3E || b == 0x64 || b == 0x65)
            d->seg = b;
        else
            break;
        if (++i >= 14)
            return -1;
    }
    d->nprefix = i;

    op = code[i++];
    if (op == 0x0F)
    {
        uint32_t op2 = code[i++];
        if (op2 == 0x38)
        {
            d->map = 2;
            d->op = code[i++];
            hasModrm = 1;
        }
        else if (op2 == 0x3A)
        {
            d->map = 3;
            d->op = code[i++];
            hasModrm = 1;
            immSize = 1;
        }
        else
        {
            d->map = 1;
            d->op = op2;
            hasModrm = !TestBit(kNoModrm2, op2);
            if (TestBit(kImm8_2, op2))
                immSize = 1;
            else if (op2 >= 0x80 && op2 <= 0x8F)
                immSize = d->opsize16 ? 2 : 4; // Jcc rel
        }
    }
    else
    {
        d->map = 0;
        d->op = op;
        hasModrm = TestBit(kModrm1, op);
        // Immediates of the one-byte map.
        if (op < 0x40 && (op & 7) == 4)
            immSize = 1; // ALU AL,Ib
        else if (op < 0x40 && (op & 7) == 5)
            immSize = d->opsize16 ? 2 : 4; // ALU eAX,Iz
        else
        {
            switch (op)
            {
            case 0x68: case 0x69: case 0x81: case 0xA9: case 0xC7: case 0xE8: case 0xE9:
                immSize = d->opsize16 ? 2 : 4;
                break;
            case 0x6A: case 0x6B: case 0x80: case 0x82: case 0x83: case 0xA8: case 0xC0: case 0xC1:
            case 0xC6: case 0xCD: case 0xD4: case 0xD5: case 0xE0: case 0xE1: case 0xE2: case 0xE3:
            case 0xE4: case 0xE5: case 0xE6: case 0xE7: case 0xEB:
                immSize = 1;
                break;
            case 0xC2: case 0xCA:
                immSize = 2;
                break;
            case 0xC8:
                immSize = 3;
                break;
            case 0x9A: case 0xEA:
                immSize = d->opsize16 ? 4 : 6;
                break;
            default:
                if (op >= 0x70 && op <= 0x7F)
                    immSize = 1; // Jcc rel8
                else if (op >= 0xB0 && op <= 0xB7)
                    immSize = 1;
                else if (op >= 0xB8 && op <= 0xBF)
                    immSize = d->opsize16 ? 2 : 4;
                break;
            }
        }
        if (op >= 0xA0 && op <= 0xA3)
        {
            // MOV with a 32-bit absolute address (moffs).
            if (d->addr16)
                return -1;
            d->memKind = X86_MEM_MOFFS;
            d->memSize = (op & 1) ? OpSize(d) : 1;
            d->access = op <= 0xA1 ? X86_ACC_READ : X86_ACC_WRITE;
            d->dispOff = i;
            d->dispSize = 4;
            d->disp = (int32_t)((uint32_t)code[i] | ((uint32_t)code[i + 1] << 8) |
                                ((uint32_t)code[i + 2] << 16) | ((uint32_t)code[i + 3] << 24));
            i += 4;
        }
        else if ((op >= 0xA4 && op <= 0xA7) || (op >= 0xAA && op <= 0xAF) || (op >= 0x6C && op <= 0x6F))
        {
            static const uint8_t kStr[16] = {
                [0x4] = X86_STR_MOVS, [0x5] = X86_STR_MOVS, [0x6] = X86_STR_CMPS, [0x7] = X86_STR_CMPS,
                [0xA] = X86_STR_STOS, [0xB] = X86_STR_STOS, [0xC] = X86_STR_LODS, [0xD] = X86_STR_LODS,
                [0xE] = X86_STR_SCAS, [0xF] = X86_STR_SCAS,
            };
            d->memKind = X86_MEM_STRING;
            if (op >= 0x6C && op <= 0x6F)
                d->strKind = op <= 0x6D ? X86_STR_INS : X86_STR_OUTS;
            else
                d->strKind = kStr[op & 0xF];
            d->memSize = (op & 1) ? OpSize(d) : 1;
            switch (d->strKind)
            {
            case X86_STR_MOVS: d->access = X86_ACC_READ | X86_ACC_WRITE; break;
            case X86_STR_STOS: case X86_STR_INS: d->access = X86_ACC_WRITE; break;
            default: d->access = X86_ACC_READ; break;
            }
        }
    }

    if (hasModrm)
    {
        uint32_t modrm = code[i];
        d->hasModrm = 1;
        d->modrmOff = i++;
        d->modrm = modrm;
        d->mod = modrm >> 6;
        d->reg = (modrm >> 3) & 7;
        d->rm = modrm & 7;
        if (d->mod != 3)
        {
            if (d->addr16)
                return -1; // 16-bit addressing: never emitted by GCC for i386
            d->memKind = X86_MEM_MODRM;
            if (d->rm == 4)
            {
                uint32_t sib = code[i++];
                d->hasSib = 1;
                d->sib = sib;
                d->scale = sib >> 6;
                d->index = ((sib >> 3) & 7) == 4 ? X86_REG_NONE : (sib >> 3) & 7;
                if ((sib & 7) == 5 && d->mod == 0)
                    d->dispSize = 4; // no base, disp32
                else
                    d->base = sib & 7;
            }
            else if (d->rm == 5 && d->mod == 0)
            {
                d->dispSize = 4; // disp32 absolute
            }
            else
            {
                d->base = d->rm;
            }
            if (d->mod == 1)
                d->dispSize = 1;
            else if (d->mod == 2)
                d->dispSize = 4;
            d->dispOff = i;
            if (d->dispSize == 1)
                d->disp = (int8_t)code[i];
            else if (d->dispSize == 4)
                d->disp = (int32_t)((uint32_t)code[i] | ((uint32_t)code[i + 1] << 8) |
                                    ((uint32_t)code[i + 2] << 16) | ((uint32_t)code[i + 3] << 24));
            i += d->dispSize;
        }
        // F6/F7 /0 and /1 (TEST) take an immediate.
        if (d->map == 0 && (op == 0xF6 || op == 0xF7) && d->reg <= 1)
            immSize = op == 0xF6 ? 1 : (d->opsize16 ? 2 : 4);
        if (d->memKind == X86_MEM_MODRM)
        {
            if (d->map == 0)
                MemoryModelOneByte(d);
            else if (d->map == 1)
                MemoryModel0F(d);
            else
                SetAccess(d, 16, X86_ACC_READ); // SSSE3/SSE4: loads (not enabled in this build)
        }
    }
    d->immOff = i;
    d->immSize = immSize;
    i += immSize;
    if (i > 15)
        return -1;
    d->len = i;
    return 0;
}

uint32_t X86EffectiveAddress(const struct X86Insn *d, const uint32_t regs[8])
{
    uint32_t ea = (uint32_t)d->disp;
    if (d->memKind == X86_MEM_MOFFS)
        return ea;
    if (d->base != X86_REG_NONE)
        ea += regs[d->base];
    if (d->index != X86_REG_NONE)
        ea += regs[d->index] << d->scale;
    return ea;
}
