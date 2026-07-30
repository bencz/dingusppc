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

/** @file A compiled block, as the cache and the execution loop see it.

    This is the only thing those two know about a backend's output, which is
    what keeps them free of host detail.
 */

#ifndef PPC_JIT_BLOCK_H
#define PPC_JIT_BLOCK_H

#include <cinttypes>

namespace dppc_jit {

class Backend;
struct JitBlock;
struct ChainRef;

/** Runs compiled code, starting with this block.

    On entry ppc_state.pc holds the address of the first instruction. On
    return it holds the address of the guest instruction that did not run,
    and exec_flags carries whatever the guest instructions raised, exactly as
    the interpreter leaves them after one instruction.

    The block accounts for its own retired cycles, including any it owes at a
    sync point in the middle. Nothing outside has to know how long it was.

    One call does not mean one block. A native backend enters its own frame
    here and stays in it, running block after block through its dispatch stub
    for as long as nothing needs the caller's attention, so this returns at a
    guest exception, at a block no backend took, or at the goal of an `until`
    run. That is the whole point of the shared frame: a prologue and epilogue
    per block cost more than the block.

    No C++ exception may leave this call. A native backend arranges that with
    the wrappers in jitruntime.h; a frame it emitted has no unwind
    information, so an exception crossing it is undefined behaviour */
typedef void (*BlockEntry)(const JitBlock* blk);

struct JitBlock {
    uint32_t    virt_addr;  // where the block starts
    uint32_t    phys_addr;  // the invalidation key, see ppccodecache.h
    uint32_t    mode;       // ppc_jit_mode() the block was translated under
    uint32_t    byte_size;  // guest bytes covered on the entry's page
    int32_t     end_off;    // one past the last instruction, the fall off PC

    /** The second invalidation range a cross page walk through registers,
        zero size when the block never left its page */
    uint32_t    second_phys;
    uint32_t    second_size;

    uint32_t    insn_count; // guest instructions covered
    uint8_t     end_reason; // BlockEnd, for diagnostics
    BlockEntry  entry;
    void*       payload;    // backend private, released by Backend::release

    /** Where generated code jumps to run this block, or nullptr when the
        block can only be entered through `entry`.

        The threaded backend leaves it null: its blocks are ordinary C++ and
        there is no frame to share, so reaching one ends the native run and
        the loop in ppcjit.cpp picks it up. rt_dispatch is what reads this */
    void*       code;

    /** Who produced it. A native backend declining a block sends it to the
        threaded one, so within a single run blocks can come from either and
        each has to go home to be released */
    Backend*    owner;

    /** Entries seen while the block is still threaded. A block is only
        emitted once this crosses the promotion threshold, which is what
        keeps code that runs once, of which a booting system has megabytes,
        from ever paying for the emitter */
    uint32_t    heat;

    /** Chain slots elsewhere that jump straight into this block, so they can
        all go back to resolving the moment it dies. The entries live in the
        slots themselves, see ChainRef in jitruntime.h; this is just the head
        of their list */
    ChainRef*   chain_in;

    /** Neighbours in the list of blocks whose chain_in is nonempty, which is
        what unbinding everything walks instead of every block there is */
    JitBlock*   chained_prev;
    JitBlock*   chained_next;
};

/** Lookup key. The physical address alone identifies the memory, the mode
    identifies the rules the instructions were decoded under */
inline uint64_t block_key(uint32_t phys_addr, uint32_t mode) {
    return (uint64_t(mode) << 32) | phys_addr;
}

} // namespace dppc_jit

#endif // PPC_JIT_BLOCK_H
