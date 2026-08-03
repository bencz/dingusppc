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

/** @file Public face of the block translator.

    Everything the rest of the emulator needs to know about the JIT lives
    here. The execution loops in ppcexec.cpp call ppc_jit_exec_inner instead
    of their own inner loop while the JIT is on; nothing else changes.

    The JIT is off unless something turns it on, and there is no command line
    flag for it yet. Only the tests enable it so far.
 */

#ifndef PPC_JIT_H
#define PPC_JIT_H

#include "ppcemu.h"

#include <cinttypes>

/** MSR bits a translation is allowed to depend on.

    Blocks are keyed by physical address and by these bits together. The
    opcode table itself changes with MSR[FP] (see ppc_msr_did_change), what is
    privileged changes with MSR[PR], and which TLB an emitted memory access
    would look at changes with MSR[IR] and MSR[DR]. A block translated under
    one combination says nothing about another */
constexpr uint32_t PPC_JIT_MODE_MASK = MSR::FP | MSR::PR | MSR::IR | MSR::DR;

inline uint32_t ppc_jit_mode() {
    return ppc_state.msr & PPC_JIT_MODE_MASK;
}

/** Which execution loop is asking.

    `debug` is missing on purpose. It compares the PC against a whole region
    after every instruction, and there is no cheap way to give a block that
    granularity. Debugging stays on the interpreter.

    `until` compares against one address, which is affordable: a block whose
    range contains the goal gets stepped through on the interpreter instead of
    run whole, so the PC is observed at the same points either way */
enum class JitExecType {
    run,    // run until power goes off
    until,  // run until the PC reaches a given address
};

/** Read by the outer loops on every iteration, so keep it a plain load */
extern bool ppc_jit_enabled;

/** Set by the --jit command line flag before the machine comes up, read by
    ppc_cpu_init. The DPPC_JIT environment variable overrides it either way,
    since that is the diagnostic switch and it can also say `threaded` or 0 */
extern bool ppc_jit_requested;

inline bool ppc_jit_is_enabled() {
    return ppc_jit_enabled;
}

enum class JitBackend {
    /** Emitter for this host, with the ordinary interpreter handling cold
        or declined instructions. What a normal run wants */
    automatic,

    /** Threaded only, ignoring any emitter. The tests use it to run the same
        guest code both ways and compare */
    threaded,
};

/** Brings up a backend and starts translating. Returns false when nothing
    could be brought up, in which case the JIT stays off and everything keeps
    running on the interpreter.

    The threaded backend executes a block by calling the interpreter helpers
    in sequence. It is not the automatic fallback and is not expected to be
    faster; it remains available explicitly for differential tests */
bool ppc_jit_enable(JitBackend choice = JitBackend::automatic);

/** Drops every block and goes back to the interpreter */
void ppc_jit_disable();

/** Drops every block, keeping the JIT on */
void ppc_jit_flush();

/** Block executing counterpart of ppc_exec_inner. Runs until power goes off,
    or until the PC reaches goal_addr when type is `until`.

    Untranslatable instructions and modes fall back to the interpreter one
    instruction at a time, so this always makes progress */
void ppc_jit_exec_inner(JitExecType type, uint32_t goal_addr);

/** How many blocks the backend is currently holding. For the tests */
unsigned ppc_jit_num_blocks();

/** Blocks compiled since the JIT came up, by which backend took them. Tells
    the tests whether the emitter is really doing the work or quietly
    declining everything */
unsigned ppc_jit_native_compiles();
unsigned ppc_jit_threaded_compiles();

/** Number of chain slots currently bound to live blocks. The registry already
    maintains this value, so exposing it to tests adds no execution cost. */
uint64_t ppc_jit_bound_chains();

/** Name of the backend in use, or nullptr while the JIT is off */
const char* ppc_jit_backend_name();

#endif // PPC_JIT_H
