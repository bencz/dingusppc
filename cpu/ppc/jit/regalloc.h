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

/** @file Block local register allocation.

    Nothing here is implemented yet, and that is deliberate. This file is the
    interface the first emitter will fill in; the IR currently carries no
    values for it to allocate, so an implementation now would be guesswork
    with nothing to check it against.

    What is already settled is the split. Liveness is pure IR analysis and is
    shared. Allocation is not: x86-64 wants CL for a variable shift and
    EDX:EAX for a division, AArch64 has no equivalent constraint, so the host
    side arrives through AbiDesc for the register file and through a per
    instruction hook for the rest.

    One rule is not negotiable and belongs written down here. Any call out of
    generated code can raise a guest exception or unwind, and at that point
    the execution loop reads guest state out of ppc_state. So every guest
    register the block has been keeping in a host register has to be written
    back before such a call, not merely at the block exit. A stock allocator
    from a code generation library has no way to know that, which is one of
    the reasons this one is ours.
 */

#ifndef PPC_JIT_REGALLOC_H
#define PPC_JIT_REGALLOC_H

#include "abi.h"
#include "jitir.h"

#include <cinttypes>
#include <vector>

namespace dppc_jit {

/** Half open interval of IR instruction indices over which a value is live */
struct LiveRange {
    uint32_t vreg;
    uint32_t first;
    uint32_t last;
};

/** Where a value ended up */
struct Assignment {
    uint32_t vreg;
    int16_t  host_reg;   // negative when spilled
    int16_t  spill_slot; // negative when it lives in a register
};

/** What a single IR instruction demands of the host register file. The
    backend answers for its own architecture; everything above this line stays
    architecture neutral */
class RegConstraints {
public:
    virtual ~RegConstraints() = default;

    /** Registers this instruction destroys regardless of the allocation, the
        way an x86 division destroys EDX */
    virtual uint32_t clobbers(const IRInsn& insn) const = 0;

    /** Registers a given operand is restricted to, or all usable registers
        when it is not restricted */
    virtual uint32_t operand_choices(const IRInsn& insn, unsigned operand) const = 0;
};

/** Liveness over the IR, host independent */
void compute_live_ranges(const IRBlock& ir, std::vector<LiveRange>& out);

/** Linear scan over the ranges, driven by the ABI register file and the
    backend's constraints */
void allocate_registers(const IRBlock& ir, const std::vector<LiveRange>& ranges,
                        const AbiDesc& abi, const RegConstraints& constraints,
                        std::vector<Assignment>& out);

} // namespace dppc_jit

#endif // PPC_JIT_REGALLOC_H
