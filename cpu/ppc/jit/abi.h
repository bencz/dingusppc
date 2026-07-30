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

/** @file Calling convention, described so a backend does not have to hard
    code one.

    Architecture and ABI are independent axes. There is one backend per host
    architecture and it gets one of these injected, so the System V and the
    Microsoft x64 builds are the same emitter with a different descriptor, and
    likewise for AAPCS64 and the Apple variant.

    Register numbers are whatever the architecture's own encoding is, so the
    masks below mean nothing without the backend that supplies them.
 */

#ifndef PPC_JIT_ABI_H
#define PPC_JIT_ABI_H

#include <cinttypes>

namespace dppc_jit {

/** Consumed by the register allocator, which needs to know what it may use
    and what it has to hand back, and by the block prologue, which needs to
    know what the stack has to look like at a call site */
struct AbiDesc {
    const char* name;

    /** Free to clobber across a call, so the allocator can use them without
        saving, but has to spill anything live before a helper call */
    uint32_t volatile_gprs;

    /** Preserved across a call, so the allocator can pin a guest register in
        one for the whole block, at the cost of saving it in the prologue */
    uint32_t callee_saved_gprs;

    /** Never touched: the stack pointer, the frame pointer where the platform
        requires one, and the platform register */
    uint32_t reserved_gprs;

    const uint8_t* int_arg_regs;     // in order
    uint8_t        num_int_arg_regs;
    uint8_t        int_ret_reg;

    /** Alignment the stack pointer needs at the point a call is made */
    uint16_t stack_alignment;

    /** Bytes above the return address the callee owns, 32 on Microsoft x64
        and 0 everywhere else we care about */
    uint16_t shadow_space;
};

/** Registers the allocator is allowed to hand out */
inline uint32_t abi_usable_gprs(const AbiDesc& abi) {
    return (abi.volatile_gprs | abi.callee_saved_gprs) & ~abi.reserved_gprs;
}

/** Picks the descriptor matching the host this build targets */
const AbiDesc& abi_for_host();

} // namespace dppc_jit

#endif // PPC_JIT_ABI_H
