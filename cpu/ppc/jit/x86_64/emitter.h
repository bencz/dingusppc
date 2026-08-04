/*
DingusPPC - The Experimental PowerPC Macintosh emulator
Copyright (C) 2018-26 The DingusPPC Development Team
          (See CREDITS.MD for more details)

(You may also contact divingkxt or powermax2286 on Discord)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

/** @file x86-64 instruction encoder.

    Written by hand and kept to the forms the emitter actually uses. It grows
    an instruction at a time as opcode coverage grows, rather than covering
    the architecture up front.

    Encoding goes into an ordinary vector, not straight into code memory. The
    output is position independent, every jump inside it being relative and
    every address inside it being an immediate, so the finished bytes are
    copied wherever they end up. That saves having to guess a block's size
    before emitting it.

    Register numbers are the architecture's own, so they line up with what
    AbiDesc carries: 0 rax, 1 rcx, 2 rdx, 3 rbx, 4 rsp, 5 rbp, 6 rsi, 7 rdi,
    8 to 15 r8 to r15.
 */

#ifndef PPC_JIT_X86_64_EMITTER_H
#define PPC_JIT_X86_64_EMITTER_H

#include <cinttypes>
#include <cstddef>
#include <vector>

namespace dppc_jit {

enum X64Gpr : uint8_t {
    RAX = 0, RCX, RDX, RBX, RSP, RBP, RSI, RDI,
    R8, R9, R10, R11, R12, R13, R14, R15
};

/** Low nibble of the condition, shared by jcc, setcc and cmovcc */
enum class X64Cond : uint8_t {
    Overflow   = 0x0, // OF set, the signed overflow of what just ran
    Below      = 0x2, // unsigned <, which is also CF set
    AboveEqual = 0x3, // unsigned >=, which is also CF clear
    Equal      = 0x4, // ZF set
    NotEqual   = 0x5, // ZF clear
    BelowEqual = 0x6, // unsigned <=
    Above      = 0x7, // unsigned >
    Less       = 0xC, // signed <
    GreaterEqual = 0xD, // signed >=
    LessEqual  = 0xE, // signed <=
    Greater    = 0xF, // signed >
};

class X64Emitter {
public:
    /** Forward reference to somewhere not emitted yet */
    typedef int32_t Label;

    X64Emitter();

    const uint8_t* bytes() const { return this->buf.data(); }
    size_t size() const { return this->buf.size(); }
    void clear();

    /** Resolves every jump against the label it names. Must run before the
        bytes are used for anything. Returns false when a label was jumped to
        and never bound, which is a bug in the emitter rather than something
        a guest program can cause */
    bool finalize();

    /** Second and last pass, run once the bytes have been copied to the
        address they will execute from.

        Only jmp_abs needs it. A jump reaching outside this buffer cannot be
        encoded while the buffer floats, because rel32 is measured from where
        the instruction ends up. Returns false when a target turned out to be
        further than rel32 reaches, which the caller has to treat as the block
        being uncompilable rather than as something to ignore */
    bool relocate(uint8_t* final_base);

    Label new_label();
    void bind(Label label);

    // stack
    void push(X64Gpr reg);
    void pop(X64Gpr reg);
    void nop8();
    void sub_rsp(int32_t bytes);
    void add_rsp(int32_t bytes);

    // moves
    void mov_reg_imm64(X64Gpr dst, uint64_t imm);   // movabs, 64 bit
    void mov_reg_imm32(X64Gpr dst, uint32_t imm);   // zero extends
    void mov_reg_reg32(X64Gpr dst, X64Gpr src);
    void mov_reg_mem32(X64Gpr dst, X64Gpr base, int32_t disp);
    void mov_mem_reg32(X64Gpr base, int32_t disp, X64Gpr src);
    void movbe_reg_mem32(X64Gpr dst, X64Gpr base, int32_t disp);
    void movbe_mem_reg32(X64Gpr base, int32_t disp, X64Gpr src);
    void movbe_mem_reg16(X64Gpr base, int32_t disp, X64Gpr src);
    void lea_reg_mem(X64Gpr dst, X64Gpr base, int32_t disp);
    void lea_reg_reg32(X64Gpr dst, X64Gpr base, X64Gpr index); // dst = base + index, FLAGS untouched

    // arithmetic and tests
    void add_reg_reg32(X64Gpr dst, X64Gpr src);   // dst += src
    void sub_reg_reg32(X64Gpr dst, X64Gpr src);   // dst -= src
    void adc_reg_reg32(X64Gpr dst, X64Gpr src);   // dst += src + CF
    void imul_reg_reg32(X64Gpr dst, X64Gpr src);  // dst *= src, OF on overflow
    void imul_reg_reg_imm32(X64Gpr dst, X64Gpr src, uint32_t imm);
    void not_reg32(X64Gpr dst);                   // leaves the flags alone
    void neg_reg32(X64Gpr dst);                   // dst = -dst
    void setcc_reg8(X64Cond cond, X64Gpr dst);    // low byte only
    void and_reg_reg32(X64Gpr dst, X64Gpr src);
    void or_reg_reg32(X64Gpr dst, X64Gpr src);
    void or_reg_mem32(X64Gpr dst, X64Gpr base, int32_t disp);
    void and_reg_imm32(X64Gpr dst, uint32_t imm);
    void or_reg_imm32(X64Gpr dst, uint32_t imm);
    void xor_reg_imm32(X64Gpr dst, uint32_t imm);
    void rol_reg_imm8(X64Gpr dst, uint8_t sh);
    void rol_reg16_imm8(X64Gpr dst, uint8_t sh);
    void shr_reg_imm8(X64Gpr dst, uint8_t sh);
    void shl_reg_imm8(X64Gpr dst, uint8_t sh);
    void sar_reg_imm8(X64Gpr dst, uint8_t sh);
    void rol_reg_cl32(X64Gpr dst);
    void shl_reg_cl32(X64Gpr dst);
    void shr_reg_cl32(X64Gpr dst);
    void sar_reg_cl32(X64Gpr dst);
    void bsr_reg_reg32(X64Gpr dst, X64Gpr src);
    void cdq();
    void div_reg32(X64Gpr divisor);
    void idiv_reg32(X64Gpr divisor);
    void bswap_reg32(X64Gpr dst);
    void test_reg_imm32(X64Gpr dst, uint32_t imm);
    void cmp_mem_reg32(X64Gpr base, int32_t disp, X64Gpr src);

    // 64 bit forms, for host pointers: TLB entries and the host address a
    // guest address maps to
    void mov_reg64_reg64(X64Gpr dst, X64Gpr src);
    void shr_reg64_imm8(X64Gpr dst, uint8_t sh);
    void imul_reg64_reg64(X64Gpr dst, X64Gpr src); // full 64 bit product
    void movsxd_reg64_reg32(X64Gpr dst, X64Gpr src); // sign extend low word
    void mov_reg64_mem(X64Gpr dst, X64Gpr base, int32_t disp);
    void mov_mem_reg64(X64Gpr base, int32_t disp, X64Gpr src);
    void btc_reg64_imm8(X64Gpr dst, uint8_t bit);
    void btr_reg64_imm8(X64Gpr dst, uint8_t bit);
    void bts_reg64_imm8(X64Gpr dst, uint8_t bit);
    void add_reg64_mem(X64Gpr dst, X64Gpr base, int32_t disp);
    void add_reg64_reg64(X64Gpr dst, X64Gpr src);
    void movzx_reg8(X64Gpr dst, X64Gpr src);      // zero extend low byte
    void movzx_reg8_mem(X64Gpr dst, X64Gpr base, int32_t disp);
    void movzx_reg16_mem(X64Gpr dst, X64Gpr base, int32_t disp);
    void mov_mem_reg8(X64Gpr base, int32_t disp, X64Gpr src);
    void mov_mem_reg16(X64Gpr base, int32_t disp, X64Gpr src);
    void cmp_reg_imm32(X64Gpr dst, uint32_t imm);
    void cmp_reg_reg32(X64Gpr dst, X64Gpr src);
    void cmov_reg_reg32(X64Cond cond, X64Gpr dst, X64Gpr src);
    void movsx_reg8(X64Gpr dst, X64Gpr src);      // sign extend low byte
    void movsx_reg16(X64Gpr dst, X64Gpr src);     // sign extend low word
    void add_reg_imm32(X64Gpr dst, uint32_t imm);
    void add_reg_imm32_flags(X64Gpr dst, uint32_t imm); // never elides zero
    void adc_reg_imm32(X64Gpr dst, uint32_t imm);
    void inc_reg32(X64Gpr dst);
    void xor_reg_reg32(X64Gpr dst, X64Gpr src);
    void test_reg8_self(X64Gpr reg);
    void test_reg64_self(X64Gpr reg);   // all 64 bits, for a returned pointer
    void cmp_mem_imm8(X64Gpr base, int32_t disp, int8_t imm);
    void mov_mem_imm32(X64Gpr base, int32_t disp, uint32_t imm);
    void and_mem_imm32(X64Gpr base, int32_t disp, uint32_t imm);
    void or_mem_imm32(X64Gpr base, int32_t disp, uint32_t imm);
    void test_mem_imm32(X64Gpr base, int32_t disp, uint32_t imm);
    void dec_mem32(X64Gpr base, int32_t disp);

    // 64 bit forms, for the retired instruction counter and its deadline
    void add_mem64_imm32(X64Gpr base, int32_t disp, uint32_t imm);
    void cmp_reg64_mem(X64Gpr dst, X64Gpr base, int32_t disp);
    void cmp_mem8_imm8(X64Gpr base, int32_t disp, uint8_t imm);

    // control flow
    void call_reg(X64Gpr reg);
    void jcc(X64Cond cond, Label label);
    void jmp(Label label);

    /** Tail call to something already emitted elsewhere in code memory.
        Resolved by relocate, not by finalize */
    void jmp_abs(const void* target);

    /** Indirect jump through an absolute 8 byte slot, `jmp qword [rip+disp]`.
        The slot lives in the data tail of code memory, so rewriting it
        redirects every block that jumps through it without touching a byte
        of code. Resolved by relocate, like jmp_abs */
    void jmp_mem_abs(const void* slot);

    /** RIP relative data reads against a fixed address, for the fields of a
        chain slot. Resolved by relocate, like jmp_abs */
    void mov_reg64_mem_abs(X64Gpr dst, const void* addr);
    void cmp_reg_mem32_abs(X64Gpr reg, const void* addr);
    void mov_mem64_abs_reg(const void* addr, X64Gpr src); // the matching write

    void jmp_reg(X64Gpr reg);
    void ret();

private:
    typedef struct Fixup {
        size_t  at;    // offset of the rel32 field
        Label   label;
    } Fixup;

    typedef struct AbsFixup {
        size_t      at;      // offset of the rel32 field
        const void* target;
    } AbsFixup;

    void emit8(uint8_t b);
    void emit32(uint32_t v);
    void emit64(uint64_t v);
    void rex(bool w, uint8_t reg, uint8_t base, uint8_t index = 0);
    void modrm_mem(uint8_t reg, uint8_t base, int32_t disp);
    void modrm_reg(uint8_t reg, uint8_t rm);
    void alu_reg_imm32(uint8_t ext, X64Gpr dst, uint32_t imm);

    std::vector<uint8_t>  buf;
    std::vector<size_t>   labels; // offset, or SIZE_MAX while unbound
    std::vector<Fixup>    fixups;
    std::vector<AbsFixup> abs_fixups;
};

} // namespace dppc_jit

#endif // PPC_JIT_X86_64_EMITTER_H
