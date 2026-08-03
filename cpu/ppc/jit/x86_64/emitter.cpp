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

/** @file x86-64 instruction encoder. */

#include "emitter.h"

#include <cstring>
#include <limits>

namespace dppc_jit {

namespace {

constexpr size_t UNBOUND = std::numeric_limits<size_t>::max();

inline bool fits_int8(int32_t v) {
    return v >= -128 && v <= 127;
}

} // namespace

X64Emitter::X64Emitter() {
    this->buf.reserve(512);
}

void X64Emitter::clear() {
    this->buf.clear();
    this->labels.clear();
    this->fixups.clear();
    this->abs_fixups.clear();
}

void X64Emitter::emit8(uint8_t b) {
    this->buf.push_back(b);
}

void X64Emitter::emit32(uint32_t v) {
    // the host is the target here, so the in memory order is already right
    const uint8_t* p = reinterpret_cast<const uint8_t*>(&v);
    this->buf.insert(this->buf.end(), p, p + 4);
}

void X64Emitter::emit64(uint64_t v) {
    const uint8_t* p = reinterpret_cast<const uint8_t*>(&v);
    this->buf.insert(this->buf.end(), p, p + 8);
}

/** REX is only emitted when something needs it: a 64 bit operand, or one of
    the registers added by the 64 bit extension */
void X64Emitter::rex(bool w, uint8_t reg, uint8_t base, uint8_t index) {
    const uint8_t bits = (w ? 0x8 : 0) | ((reg >> 3) << 2) | ((index >> 3) << 1) | (base >> 3);
    if (bits) {
        this->emit8(0x40 | bits);
    }
}

/** ModRM plus whatever SIB and displacement [base + disp] needs.

    Two encodings force a shape rather than let one be chosen. rm 100 means a
    SIB byte follows instead of naming a register, so rsp and r12 always take
    one. rm 101 with mod 00 means RIP relative rather than a register, so rbp
    and r13 always carry a displacement even when it is zero */
void X64Emitter::modrm_mem(uint8_t reg, uint8_t base, int32_t disp) {
    const uint8_t r  = reg & 7;
    const uint8_t b  = base & 7;
    const bool needs_sib  = (b == 4);
    const bool needs_disp = (disp != 0) || (b == 5);

    uint8_t mod;
    if (!needs_disp) {
        mod = 0;
    } else if (fits_int8(disp)) {
        mod = 1;
    } else {
        mod = 2;
    }

    this->emit8(uint8_t((mod << 6) | (r << 3) | (needs_sib ? 4 : b)));
    if (needs_sib) {
        this->emit8(uint8_t((0 << 6) | (4 << 3) | b)); // scale 1, no index
    }
    if (mod == 1) {
        this->emit8(uint8_t(int8_t(disp)));
    } else if (mod == 2) {
        this->emit32(uint32_t(disp));
    }
}

void X64Emitter::modrm_reg(uint8_t reg, uint8_t rm) {
    this->emit8(uint8_t(0xC0 | ((reg & 7) << 3) | (rm & 7)));
}

X64Emitter::Label X64Emitter::new_label() {
    this->labels.push_back(UNBOUND);
    return Label(this->labels.size() - 1);
}

void X64Emitter::bind(Label label) {
    this->labels[size_t(label)] = this->buf.size();
}

bool X64Emitter::finalize() {
    for (const Fixup& f : this->fixups) {
        const size_t target = this->labels[size_t(f.label)];
        if (target == UNBOUND) {
            return false;
        }
        // rel32 counts from the end of the instruction, which is where the
        // displacement field itself ends
        const int32_t rel = int32_t(int64_t(target) - int64_t(f.at + 4));
        std::memcpy(&this->buf[f.at], &rel, 4);
    }
    this->fixups.clear();
    return true;
}

bool X64Emitter::relocate(uint8_t* final_base) {
    for (const AbsFixup& f : this->abs_fixups) {
        const uint8_t* end = final_base + f.at + 4;
        const int64_t rel  = int64_t(uintptr_t(f.target)) - int64_t(uintptr_t(end));
        if (rel < INT32_MIN || rel > INT32_MAX) {
            return false;
        }
        const int32_t rel32 = int32_t(rel);
        std::memcpy(final_base + f.at, &rel32, 4);
    }
    this->abs_fixups.clear();
    return true;
}

void X64Emitter::push(X64Gpr reg) {
    this->rex(false, 0, uint8_t(reg));
    this->emit8(uint8_t(0x50 + (reg & 7)));
}

void X64Emitter::pop(X64Gpr reg) {
    this->rex(false, 0, uint8_t(reg));
    this->emit8(uint8_t(0x58 + (reg & 7)));
}

void X64Emitter::nop8() {
    // Intel's canonical eight-byte NOP: one decoded instruction, no state.
    static constexpr uint8_t bytes[] = {0x0F, 0x1F, 0x84, 0x00,
                                        0x00, 0x00, 0x00, 0x00};
    for (uint8_t byte : bytes) {
        this->emit8(byte);
    }
}

void X64Emitter::sub_rsp(int32_t bytes) {
    if (bytes == 0) {
        return;
    }
    this->rex(true, 0, RSP);
    if (fits_int8(bytes)) {
        this->emit8(0x83);
        this->modrm_reg(5, RSP); // /5 is sub
        this->emit8(uint8_t(int8_t(bytes)));
    } else {
        this->emit8(0x81);
        this->modrm_reg(5, RSP);
        this->emit32(uint32_t(bytes));
    }
}

void X64Emitter::add_rsp(int32_t bytes) {
    if (bytes == 0) {
        return;
    }
    this->rex(true, 0, RSP);
    if (fits_int8(bytes)) {
        this->emit8(0x83);
        this->modrm_reg(0, RSP); // /0 is add
        this->emit8(uint8_t(int8_t(bytes)));
    } else {
        this->emit8(0x81);
        this->modrm_reg(0, RSP);
        this->emit32(uint32_t(bytes));
    }
}

void X64Emitter::mov_reg_imm64(X64Gpr dst, uint64_t imm) {
    this->rex(true, 0, uint8_t(dst));
    this->emit8(uint8_t(0xB8 + (dst & 7)));
    this->emit64(imm);
}

void X64Emitter::mov_reg_imm32(X64Gpr dst, uint32_t imm) {
    // 32 bit destinations zero the upper half, which is what we want for a
    // guest value and saves the REX.W
    this->rex(false, 0, uint8_t(dst));
    this->emit8(uint8_t(0xB8 + (dst & 7)));
    this->emit32(imm);
}

void X64Emitter::mov_reg_reg32(X64Gpr dst, X64Gpr src) {
    this->rex(false, uint8_t(src), uint8_t(dst));
    this->emit8(0x89);
    this->modrm_reg(uint8_t(src), uint8_t(dst));
}

void X64Emitter::mov_reg_mem32(X64Gpr dst, X64Gpr base, int32_t disp) {
    this->rex(false, uint8_t(dst), uint8_t(base));
    this->emit8(0x8B);
    this->modrm_mem(uint8_t(dst), uint8_t(base), disp);
}

void X64Emitter::mov_mem_reg32(X64Gpr base, int32_t disp, X64Gpr src) {
    this->rex(false, uint8_t(src), uint8_t(base));
    this->emit8(0x89);
    this->modrm_mem(uint8_t(src), uint8_t(base), disp);
}

void X64Emitter::lea_reg_mem(X64Gpr dst, X64Gpr base, int32_t disp) {
    this->rex(false, uint8_t(dst), uint8_t(base));
    this->emit8(0x8D);
    this->modrm_mem(uint8_t(dst), uint8_t(base), disp);
}

/** The flag preserving register add: lea dst, [base + index]. Used where an
    or of disjoint pieces must not scratch the FLAGS a compare left behind */
void X64Emitter::lea_reg_reg32(X64Gpr dst, X64Gpr base, X64Gpr index) {
    uint8_t b = uint8_t(base);
    uint8_t x = uint8_t(index);
    if (x == 4) { // rsp cannot be an index; the sum is commutative
        const uint8_t t = b; b = x; x = t;
    }
    this->rex(false, uint8_t(dst), b, x);
    this->emit8(0x8D);
    // rm 100 selects the SIB; an rbp or r13 base has to carry a disp8 of 0
    const bool needs_disp = (b & 7) == 5;
    this->emit8(uint8_t(((needs_disp ? 1 : 0) << 6) | ((uint8_t(dst) & 7) << 3) | 4));
    this->emit8(uint8_t(((x & 7) << 3) | (b & 7)));
    if (needs_disp) {
        this->emit8(0);
    }
}

/** ALU forms with an immediate share their ModRM extension: /0 add, /1 or,
    /4 and, /6 xor, /7 cmp. 0x83 sign extends its byte, so it is both shorter
    and exactly equivalent whenever the 32-bit immediate is a signed byte. */
void X64Emitter::alu_reg_imm32(uint8_t ext, X64Gpr dst, uint32_t imm) {
    this->rex(false, 0, uint8_t(dst));
    if (fits_int8(int32_t(imm))) {
        this->emit8(0x83);
        this->modrm_reg(ext, uint8_t(dst));
        this->emit8(uint8_t(int8_t(imm)));
    } else {
        this->emit8(0x81);
        this->modrm_reg(ext, uint8_t(dst));
        this->emit32(imm);
    }
}

void X64Emitter::add_reg_reg32(X64Gpr dst, X64Gpr src) {
    this->rex(false, uint8_t(src), uint8_t(dst));
    this->emit8(0x01);
    this->modrm_reg(uint8_t(src), uint8_t(dst));
}

void X64Emitter::sub_reg_reg32(X64Gpr dst, X64Gpr src) {
    this->rex(false, uint8_t(src), uint8_t(dst));
    this->emit8(0x29);
    this->modrm_reg(uint8_t(src), uint8_t(dst));
}

void X64Emitter::adc_reg_reg32(X64Gpr dst, X64Gpr src) {
    this->rex(false, uint8_t(src), uint8_t(dst));
    this->emit8(0x11);
    this->modrm_reg(uint8_t(src), uint8_t(dst));
}

void X64Emitter::not_reg32(X64Gpr dst) {
    this->rex(false, 0, uint8_t(dst));
    this->emit8(0xF7);
    this->modrm_reg(2, uint8_t(dst)); // /2 is not
}

void X64Emitter::neg_reg32(X64Gpr dst) {
    this->rex(false, 0, uint8_t(dst));
    this->emit8(0xF7);
    this->modrm_reg(3, uint8_t(dst)); // /3 is neg
}

void X64Emitter::imul_reg_reg32(X64Gpr dst, X64Gpr src) {
    // the two operand form: OF and CF say the product lost significant bits,
    // which for the signed 32 bit product is exactly mullwo's overflow
    this->rex(false, uint8_t(dst), uint8_t(src));
    this->emit8(0x0F);
    this->emit8(0xAF);
    this->modrm_reg(uint8_t(dst), uint8_t(src));
}

void X64Emitter::imul_reg64_reg64(X64Gpr dst, X64Gpr src) {
    this->rex(true, uint8_t(dst), uint8_t(src));
    this->emit8(0x0F);
    this->emit8(0xAF);
    this->modrm_reg(uint8_t(dst), uint8_t(src));
}

void X64Emitter::movsxd_reg64_reg32(X64Gpr dst, X64Gpr src) {
    this->rex(true, uint8_t(dst), uint8_t(src));
    this->emit8(0x63);
    this->modrm_reg(uint8_t(dst), uint8_t(src));
}

void X64Emitter::setcc_reg8(X64Cond cond, X64Gpr dst) {
    // the low byte of rsp, rbp, rsi and rdi is only reachable through REX,
    // and emitting one turns spl and friends on rather than ah and friends
    const bool needs_rex = (dst >= RSP && dst <= RDI);
    if (needs_rex && !(uint8_t(dst) & 8)) {
        this->emit8(0x40);
    } else {
        this->rex(false, 0, uint8_t(dst));
    }
    this->emit8(0x0F);
    this->emit8(uint8_t(0x90 + uint8_t(cond)));
    this->modrm_reg(0, uint8_t(dst));
}

void X64Emitter::and_reg_reg32(X64Gpr dst, X64Gpr src) {
    this->rex(false, uint8_t(src), uint8_t(dst));
    this->emit8(0x21);
    this->modrm_reg(uint8_t(src), uint8_t(dst));
}

void X64Emitter::or_reg_reg32(X64Gpr dst, X64Gpr src) {
    this->rex(false, uint8_t(src), uint8_t(dst));
    this->emit8(0x09);
    this->modrm_reg(uint8_t(src), uint8_t(dst));
}

void X64Emitter::or_reg_mem32(X64Gpr dst, X64Gpr base, int32_t disp) {
    this->rex(false, uint8_t(dst), uint8_t(base));
    this->emit8(0x0B);
    this->modrm_mem(uint8_t(dst), uint8_t(base), disp);
}

void X64Emitter::and_reg_imm32(X64Gpr dst, uint32_t imm) {
    this->alu_reg_imm32(4, dst, imm);
}

void X64Emitter::or_reg_imm32(X64Gpr dst, uint32_t imm) {
    if (imm == 0) {
        return;
    }
    this->alu_reg_imm32(1, dst, imm);
}

void X64Emitter::xor_reg_imm32(X64Gpr dst, uint32_t imm) {
    if (imm == 0) {
        return;
    }
    this->alu_reg_imm32(6, dst, imm);
}

void X64Emitter::rol_reg_imm8(X64Gpr dst, uint8_t sh) {
    if ((sh & 31) == 0) {
        return;
    }
    this->rex(false, 0, uint8_t(dst));
    this->emit8(0xC1);
    this->modrm_reg(0, uint8_t(dst)); // /0 is rol
    this->emit8(sh & 31);
}

void X64Emitter::shr_reg_imm8(X64Gpr dst, uint8_t sh) {
    if ((sh & 31) == 0) return;
    this->rex(false, 0, uint8_t(dst));
    this->emit8(0xC1);
    this->modrm_reg(5, uint8_t(dst)); // /5 is shr
    this->emit8(sh & 31);
}

void X64Emitter::shl_reg_imm8(X64Gpr dst, uint8_t sh) {
    if ((sh & 31) == 0) return;
    this->rex(false, 0, uint8_t(dst));
    this->emit8(0xC1);
    this->modrm_reg(4, uint8_t(dst)); // /4 is shl
    this->emit8(sh & 31);
}

void X64Emitter::bswap_reg32(X64Gpr dst) {
    this->rex(false, 0, uint8_t(dst));
    this->emit8(0x0F);
    this->emit8(uint8_t(0xC8 + (dst & 7)));
}

void X64Emitter::test_reg_imm32(X64Gpr dst, uint32_t imm) {
    this->rex(false, 0, uint8_t(dst));
    this->emit8(0xF7);
    this->modrm_reg(0, uint8_t(dst)); // /0 is test
    this->emit32(imm);
}

void X64Emitter::cmp_mem_reg32(X64Gpr base, int32_t disp, X64Gpr src) {
    this->rex(false, uint8_t(src), uint8_t(base));
    this->emit8(0x39);
    this->modrm_mem(uint8_t(src), uint8_t(base), disp);
}

void X64Emitter::mov_reg64_reg64(X64Gpr dst, X64Gpr src) {
    this->rex(true, uint8_t(src), uint8_t(dst));
    this->emit8(0x89);
    this->modrm_reg(uint8_t(src), uint8_t(dst));
}

void X64Emitter::shr_reg64_imm8(X64Gpr dst, uint8_t sh) {
    this->rex(true, 0, uint8_t(dst));
    this->emit8(0xC1);
    this->modrm_reg(5, uint8_t(dst)); // /5 is shr
    this->emit8(sh & 63);
}

void X64Emitter::mov_reg64_mem(X64Gpr dst, X64Gpr base, int32_t disp) {
    this->rex(true, uint8_t(dst), uint8_t(base));
    this->emit8(0x8B);
    this->modrm_mem(uint8_t(dst), uint8_t(base), disp);
}

void X64Emitter::add_reg64_mem(X64Gpr dst, X64Gpr base, int32_t disp) {
    this->rex(true, uint8_t(dst), uint8_t(base));
    this->emit8(0x03);
    this->modrm_mem(uint8_t(dst), uint8_t(base), disp);
}

void X64Emitter::add_reg64_reg64(X64Gpr dst, X64Gpr src) {
    this->rex(true, uint8_t(src), uint8_t(dst));
    this->emit8(0x01);
    this->modrm_reg(uint8_t(src), uint8_t(dst));
}

void X64Emitter::movzx_reg8(X64Gpr dst, X64Gpr src) {
    const bool needs_rex = (src >= RSP && src <= RDI);
    if (needs_rex && !((uint8_t(dst) | uint8_t(src)) & 8)) {
        this->emit8(0x40);
    } else {
        this->rex(false, uint8_t(dst), uint8_t(src));
    }
    this->emit8(0x0F);
    this->emit8(0xB6);
    this->modrm_reg(uint8_t(dst), uint8_t(src));
}

void X64Emitter::movzx_reg8_mem(X64Gpr dst, X64Gpr base, int32_t disp) {
    this->rex(false, uint8_t(dst), uint8_t(base));
    this->emit8(0x0F);
    this->emit8(0xB6);
    this->modrm_mem(uint8_t(dst), uint8_t(base), disp);
}

void X64Emitter::movzx_reg16_mem(X64Gpr dst, X64Gpr base, int32_t disp) {
    this->rex(false, uint8_t(dst), uint8_t(base));
    this->emit8(0x0F);
    this->emit8(0xB7);
    this->modrm_mem(uint8_t(dst), uint8_t(base), disp);
}

void X64Emitter::mov_mem_reg8(X64Gpr base, int32_t disp, X64Gpr src) {
    // the low byte of rsp, rbp, rsi and rdi needs a REX to be reachable at
    // all, and emitting one selects spl and friends rather than ah and friends
    const bool needs_rex = (src >= RSP && src <= RDI);
    if (needs_rex && !((uint8_t(src) | uint8_t(base)) & 8)) {
        this->emit8(0x40);
    } else {
        this->rex(false, uint8_t(src), uint8_t(base));
    }
    this->emit8(0x88);
    this->modrm_mem(uint8_t(src), uint8_t(base), disp);
}

void X64Emitter::mov_mem_reg16(X64Gpr base, int32_t disp, X64Gpr src) {
    this->emit8(0x66); // operand size override
    this->rex(false, uint8_t(src), uint8_t(base));
    this->emit8(0x89);
    this->modrm_mem(uint8_t(src), uint8_t(base), disp);
}

void X64Emitter::cmp_reg_imm32(X64Gpr dst, uint32_t imm) {
    this->alu_reg_imm32(7, dst, imm);
}

void X64Emitter::movsx_reg8(X64Gpr dst, X64Gpr src) {
    // the low byte of rsp, rbp, rsi and rdi is only reachable through REX,
    // and emitting one turns spl and friends on rather than ah and friends
    const bool needs_rex = (src >= RSP && src <= RDI);
    if (needs_rex && !((uint8_t(dst) | uint8_t(src)) & 8)) {
        this->emit8(0x40);
    } else {
        this->rex(false, uint8_t(dst), uint8_t(src));
    }
    this->emit8(0x0F);
    this->emit8(0xBE);
    this->modrm_reg(uint8_t(dst), uint8_t(src));
}

void X64Emitter::movsx_reg16(X64Gpr dst, X64Gpr src) {
    this->rex(false, uint8_t(dst), uint8_t(src));
    this->emit8(0x0F);
    this->emit8(0xBF);
    this->modrm_reg(uint8_t(dst), uint8_t(src));
}

void X64Emitter::add_reg_imm32(X64Gpr dst, uint32_t imm) {
    if (imm == 0) {
        return;
    }
    this->alu_reg_imm32(0, dst, imm);
}

void X64Emitter::inc_reg32(X64Gpr dst) {
    this->rex(false, 0, uint8_t(dst));
    this->emit8(0xFF);
    this->modrm_reg(0, uint8_t(dst)); // /0 is inc
}

void X64Emitter::xor_reg_reg32(X64Gpr dst, X64Gpr src) {
    this->rex(false, uint8_t(src), uint8_t(dst));
    this->emit8(0x31);
    this->modrm_reg(uint8_t(src), uint8_t(dst));
}

void X64Emitter::test_reg8_self(X64Gpr reg) {
    // the low byte of rsp, rbp, rsi and rdi is only reachable through REX,
    // and emitting one turns spl and friends on rather than ah and friends
    if (reg >= RSP) {
        this->emit8(uint8_t(0x40 | ((reg >> 3) << 2) | (reg >> 3)));
    }
    this->emit8(0x84);
    this->modrm_reg(uint8_t(reg), uint8_t(reg));
}

void X64Emitter::test_reg64_self(X64Gpr reg) {
    this->rex(true, uint8_t(reg), uint8_t(reg));
    this->emit8(0x85);
    this->modrm_reg(uint8_t(reg), uint8_t(reg));
}

void X64Emitter::cmp_mem_imm8(X64Gpr base, int32_t disp, int8_t imm) {
    this->rex(false, 0, uint8_t(base));
    this->emit8(0x83);
    this->modrm_mem(7, uint8_t(base), disp); // /7 is cmp
    this->emit8(uint8_t(imm));
}

void X64Emitter::mov_mem_imm32(X64Gpr base, int32_t disp, uint32_t imm) {
    this->rex(false, 0, uint8_t(base));
    this->emit8(0xC7);
    this->modrm_mem(0, uint8_t(base), disp);
    this->emit32(imm);
}

void X64Emitter::test_mem_imm32(X64Gpr base, int32_t disp, uint32_t imm) {
    this->rex(false, 0, uint8_t(base));
    this->emit8(0xF7);
    this->modrm_mem(0, uint8_t(base), disp); // /0 is test
    this->emit32(imm);
}

void X64Emitter::dec_mem32(X64Gpr base, int32_t disp) {
    this->rex(false, 0, uint8_t(base));
    this->emit8(0xFF);
    this->modrm_mem(1, uint8_t(base), disp); // /1 is dec
}

void X64Emitter::cmp_reg_reg32(X64Gpr dst, X64Gpr src) {
    this->rex(false, uint8_t(src), uint8_t(dst));
    this->emit8(0x39);
    this->modrm_reg(uint8_t(src), uint8_t(dst));
}

void X64Emitter::cmov_reg_reg32(X64Cond cond, X64Gpr dst, X64Gpr src) {
    this->rex(false, uint8_t(dst), uint8_t(src));
    this->emit8(0x0F);
    this->emit8(uint8_t(0x40 + uint8_t(cond)));
    this->modrm_reg(uint8_t(dst), uint8_t(src));
}

void X64Emitter::add_mem64_imm32(X64Gpr base, int32_t disp, uint32_t imm) {
    this->rex(true, 0, uint8_t(base));
    if (imm <= 0x7F) {
        this->emit8(0x83);
        this->modrm_mem(0, uint8_t(base), disp); // /0 is add
        this->emit8(uint8_t(imm));
    } else {
        this->emit8(0x81);
        this->modrm_mem(0, uint8_t(base), disp);
        this->emit32(imm);
    }
}

void X64Emitter::cmp_reg64_mem(X64Gpr dst, X64Gpr base, int32_t disp) {
    this->rex(true, uint8_t(dst), uint8_t(base));
    this->emit8(0x3B);
    this->modrm_mem(uint8_t(dst), uint8_t(base), disp);
}

void X64Emitter::cmp_mem8_imm8(X64Gpr base, int32_t disp, uint8_t imm) {
    this->rex(false, 0, uint8_t(base));
    this->emit8(0x80);
    this->modrm_mem(7, uint8_t(base), disp); // /7 is cmp
    this->emit8(imm);
}

void X64Emitter::call_reg(X64Gpr reg) {
    this->rex(false, 0, uint8_t(reg));
    this->emit8(0xFF);
    this->modrm_reg(2, uint8_t(reg)); // /2 is call
}

void X64Emitter::jcc(X64Cond cond, Label label) {
    this->emit8(0x0F);
    this->emit8(uint8_t(0x80 + uint8_t(cond)));
    this->fixups.push_back({this->buf.size(), label});
    this->emit32(0);
}

void X64Emitter::jmp(Label label) {
    this->emit8(0xE9);
    this->fixups.push_back({this->buf.size(), label});
    this->emit32(0);
}

void X64Emitter::jmp_abs(const void* target) {
    this->emit8(0xE9);
    this->abs_fixups.push_back({this->buf.size(), target});
    this->emit32(0);
}

void X64Emitter::jmp_mem_abs(const void* slot) {
    this->emit8(0xFF);
    this->emit8(0x25); // /4 with mod 00 rm 101, which is RIP relative
    this->abs_fixups.push_back({this->buf.size(), slot});
    this->emit32(0);
}

void X64Emitter::mov_reg64_mem_abs(X64Gpr dst, const void* addr) {
    this->rex(true, uint8_t(dst), 0);
    this->emit8(0x8B);
    this->emit8(uint8_t(0x05 | ((dst & 7) << 3))); // mod 00 rm 101, RIP relative
    this->abs_fixups.push_back({this->buf.size(), addr});
    this->emit32(0);
}

void X64Emitter::mov_mem64_abs_reg(const void* addr, X64Gpr src) {
    this->rex(true, uint8_t(src), 0);
    this->emit8(0x89);
    this->emit8(uint8_t(0x05 | ((src & 7) << 3))); // mod 00 rm 101, RIP relative
    this->abs_fixups.push_back({this->buf.size(), addr});
    this->emit32(0);
}

void X64Emitter::cmp_reg_mem32_abs(X64Gpr reg, const void* addr) {
    this->rex(false, uint8_t(reg), 0);
    this->emit8(0x3B);
    this->emit8(uint8_t(0x05 | ((reg & 7) << 3)));
    this->abs_fixups.push_back({this->buf.size(), addr});
    this->emit32(0);
}

void X64Emitter::jmp_reg(X64Gpr reg) {
    this->rex(false, 0, uint8_t(reg));
    this->emit8(0xFF);
    this->modrm_reg(4, uint8_t(reg)); // /4 is jmp
}

void X64Emitter::ret() {
    this->emit8(0xC3);
}

} // namespace dppc_jit
