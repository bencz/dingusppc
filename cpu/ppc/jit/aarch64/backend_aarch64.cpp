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

/** @file AArch64 emitter.

    Same shape as the x86-64 one and same state: the ABI descriptors are real,
    the emitter comes later. See backend_x86_64.cpp for why.
 */

#include "../abi.h"
#include "../backend.h"

#include <loguru.hpp>

namespace dppc_jit {

namespace {

constexpr uint32_t bit(uint8_t reg) {
    return 1u << reg;
}

/** x0 through x30, by their own numbering */
constexpr uint32_t range(uint8_t first, uint8_t last) {
    uint32_t mask = 0;
    for (uint8_t r = first; r <= last; r++) {
        mask |= bit(r);
    }
    return mask;
}

const uint8_t aapcs_args[] = {0, 1, 2, 3, 4, 5, 6, 7};

/** x16 and x17 are IP0 and IP1, which a linker veneer may overwrite at any
    call, so they count as volatile rather than usable across one. x18 is the
    platform register and is reserved outright: Apple and Windows both claim
    it, and a Linux target that does not is not worth one more register.
    x29 is the frame pointer and x30 the link register */
const AbiDesc abi_aapcs64 = {
    /* name              */ "AArch64 AAPCS64",
    /* volatile_gprs     */ range(0, 17),
    /* callee_saved_gprs */ range(19, 28),
    /* reserved_gprs     */ bit(18) | bit(29) | bit(30),
    /* int_arg_regs      */ aapcs_args,
    /* num_int_arg_regs  */ 8,
    /* int_ret_reg       */ 0,
    /* stack_alignment   */ 16,
    /* shadow_space      */ 0,
};

/** Apple's variant differs from the base one in how variadic arguments are
    passed and in reserving x18, which we reserve anyway. It is spelled out
    separately because that is the whole point of keeping the ABI a parameter,
    not because the two currently disagree on anything we use */
const AbiDesc abi_apple_arm64 = {
    /* name              */ "AArch64 Apple",
    /* volatile_gprs     */ range(0, 17),
    /* callee_saved_gprs */ range(19, 28),
    /* reserved_gprs     */ bit(18) | bit(29) | bit(30),
    /* int_arg_regs      */ aapcs_args,
    /* num_int_arg_regs  */ 8,
    /* int_ret_reg       */ 0,
    /* stack_alignment   */ 16,
    /* shadow_space      */ 0,
};

class AArch64Backend : public Backend {
public:
    explicit AArch64Backend(const AbiDesc& abi) : abi(abi) {}

    const char* name() const override {
        return this->abi.name;
    }

    JitBlock* compile(const IRBlock&) override {
        return nullptr; // minimal bring-up: the interpreter remains the fallback
    }

    void release(JitBlock*) override {}
    void release_all() override {}

private:
    const AbiDesc& abi;
};

} // namespace

const AbiDesc& abi_for_host() {
#if defined(__APPLE__)
    return abi_apple_arm64;
#else
    return abi_aapcs64;
#endif
}

std::unique_ptr<Backend> make_aarch64_backend() {
    LOG_F(INFO, "JIT: AArch64 backend has no emitter yet, %s ABI selected",
          abi_for_host().name);
    return std::unique_ptr<Backend>(new AArch64Backend(abi_for_host()));
}

} // namespace dppc_jit
