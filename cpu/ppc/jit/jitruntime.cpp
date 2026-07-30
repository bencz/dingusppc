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

/** @file The boundary generated code calls out through. */

#include "jitruntime.h"
#include "../ppcmmu.h"

namespace dppc_jit {

bool rt_call_op(PPCOpcode helper, uint32_t opcode) noexcept {
    try {
        helper(opcode);
    } catch (PPCExcUnwind&) {
        // ppc_exception_handler_unwind already set up the vector, all that is
        // left is telling the caller the instruction never finished
        return false;
    }
    return true;
}

bool rt_service_timers() noexcept {
    // only worth saying when nothing has been raised yet: anything already in
    // exec_flags has left the address in ppc_next_instruction_address itself
    const bool speak_up = exec_flags == 0;
    if (speak_up) {
        ppc_next_instruction_address = ppc_state.pc;
        exec_flags                   = EXEF_BRANCH;
    }

    try {
        g_icycles_max = ppc_process_events();
    } catch (PPCExcUnwind&) {
        // a device reached from a timer abandoned its access. The handler has
        // already run, so all that is left is telling the caller to stop
        return false;
    }

    if (speak_up && exec_flags == EXEF_BRANCH) {
        exec_flags = 0; // nothing came due, so nothing happened
    }
    return exec_flags == 0;
}

void rt_account_cycles(uint32_t retired) noexcept {
    g_icycles += retired;

    if (g_icycles <= g_icycles_max && !exec_timer) [[likely]] {
        return;
    }
    rt_service_timers();
}

bool rt_sync_cycles(uint32_t retired) noexcept {
    g_icycles += retired;

    if (g_icycles <= g_icycles_max && !exec_timer) [[likely]] {
        return exec_flags == 0;
    }
    return rt_service_timers();
}

void rt_block_end(uint32_t entry_pc, uint32_t byte_size, uint32_t retired) noexcept {
    if (!exec_flags) {
        // fell off the end, so the next guest instruction is whatever follows
        // the block. It has to be in place before the accounting below, which
        // is where an asynchronous exception can be delivered
        ppc_state.pc = entry_pc + byte_size;
    }

    rt_account_cycles(retired);
}

} // namespace dppc_jit
