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

/** @file The boundary generated code calls out through.

    Two kinds of exception meet here and they must not be confused. A guest
    exception is bookkeeping: ppc_exception_handler fills in SRR0, SRR1 and
    MSR, raises EXEF_EXCEPTION and returns, and the execution loop notices.
    A host exception is C++ unwinding, which the deep paths in the MMU and in
    device I/O use to abandon an instruction they are too far down to report
    a status through.

    A generated frame has no unwind information, so a C++ exception crossing
    one is undefined behaviour. Every call from generated code into the
    emulator therefore goes through one of these, which converts the unwind
    into a return value before it can get that far.
 */

#ifndef PPC_JIT_RUNTIME_H
#define PPC_JIT_RUNTIME_H

#include "../ppcemu.h"

#include <cinttypes>

namespace dppc_jit {

struct JitBlock;

/** One chain slot's registry entry, embedded in the slot it describes.

    The registry answers one question: when a block dies, which slots jump
    into it and have to go back to resolving. Owning the entry by the slot
    instead of allocating one per bind is what bounds the registry at the
    number of slots in emitted code, so rebinding is a move between two
    incoming lists and can never grow anything. It is also what keeps the
    bind path off cold memory: the entry shares cache lines with the slot
    fields the resolver just wrote. The pooled predecessor was measured at
    8% of the Cheetah boot storm, almost all of it one cold load per bind

    target is null while the slot is unbound, and only ppcjit.cpp touches
    any of this */
struct ChainRef {
    void**      code;     // where the target pointer lives
    const void* resolver; // what goes back there on unbind
    uint64_t*   pred;     // pred to poison on unbind, nullptr for same page
    JitBlock*   target;   // block the slot jumps into, nullptr while unbound
    ChainRef*   prev;     // neighbours on the target's incoming list
    ChainRef*   next;
};

/** Runs one interpreter helper on behalf of generated code.

    Returns true when the helper returned normally, which includes the case
    where it raised a guest exception the ordinary way; the caller finds that
    in exec_flags. Returns false when it unwound, in which case the handler
    has already picked a vector and the instruction did not retire, so the
    caller must not count it.

    Anything other than PPCExcUnwind coming out of a helper is a bug rather
    than a control transfer, and letting it take the process down beats
    unwinding through a frame the unwinder cannot describe */
bool rt_call_op(PPCOpcode helper, uint32_t opcode) noexcept;

/** Brings the retired instruction count forward and gives timers their turn.

    ppc_state.pc must already hold the address of the next guest instruction,
    and that is the whole subtlety. A timer coming due here can deliver a
    decrementer or an external interrupt, and the handler has to record where
    the guest was going. It takes that from ppc_state.pc plus four, which is
    right for the interpreter, where the PC still sits on the instruction that
    just ran, and wrong for a block, which is several instructions past it by
    the time it gets here.

    So a block says otherwise, the same way a taken branch does: the address
    goes in ppc_next_instruction_address and EXEF_BRANCH says to use it. The
    flag is put back down when nothing came due, so the ordinary case still
    leaves generated code with a clean slate.

    Getting this wrong does not show up until the guest turns interrupts on,
    and then it corrupts the return address of every one of them */
void rt_account_cycles(uint32_t retired) noexcept;

/** The rare half of settling cycles, for generated code that inlines the
    common half.

    g_icycles must already be up to date and ppc_state.pc must already be the
    instruction about to run, since that is where an exception delivered here
    resumes. Returns false when something was raised, in which case the caller
    must leave without running that instruction.

    Cannot unwind: it reaches devices through the timers */
bool rt_service_timers() noexcept;

/** Settles the cycles a block owes before an instruction that is about to
    derive virtual time from g_icycles.

    Returns false when servicing a timer raised something, in which case the
    block has to stop before running that instruction. The PC is still on the
    previous one at that point, which is what ppc_exception_handler wants */
bool rt_sync_cycles(uint32_t retired) noexcept;

/** Closes a block out: accounts for what it retired and, when nothing was
    raised, leaves the PC on whatever follows.

    `entry_pc` is the address the block was entered through and not the one it
    was translated through, because the cache is keyed by physical address and
    a page can be mapped more than once.

    Only the threaded backend still goes through this. A native block writes
    the PC itself, because it already knows both the address it was entered
    through and how far it got, and hands the count to rt_dispatch instead */
void rt_block_end(uint32_t entry_pc, uint32_t byte_size, uint32_t retired) noexcept;

/** Resolves one chain slot: finds or translates the block at ppc_state.pc,
    and when chaining is allowed binds the slot to it, so the next pass jumps
    straight through without coming here.

    Called from the resolver thunk a chained exit starts out pointing at. On
    entry the exit already advanced the entry PC register, wrote ppc_state.pc
    and settled the cycles, so `retired` would be zero and is not passed.
    Returns the code to jump to, or nullptr when the caller has to fall back
    to the dispatch stub: a block nobody compiled, the goal of an `until`
    run, or anything raised while translating.

    Binding only ever happens between two blocks and only within the guest
    page the source block lives on, which is what makes the bound jump safe
    without revalidating the MMU: the offset inside a page is the same under
    every mapping of it, and a mode change ends the source block before any
    of its exits run again. Undone by ppcjit.cpp when the target dies or an
    `until` run needs every entry observed. Nothing may unwind out of it.

    Defined in ppcjit.cpp, next to rt_dispatch, for the same reason */
typedef struct ChainSlot {
    void*    code; // stays first: the exit jumps through the slot address
    ChainRef ref;
} ChainSlot;

const void* rt_chain_resolve(ChainSlot* slot) noexcept;

/** A chain whose target is only known as a virtual address: a return or
    bcctr, whose target lives in a register, or a direct branch into another
    guest page, whose physical whereabouts depend on the mapping of the
    moment.

    The exit compares the target address against pred_va and the translation
    generation against gen before jumping through code. The generation is the
    mapping guard: mmu_itrans_generation moves whenever an instruction fetch
    could start translating differently, and a stale binding then fails the
    compare instead of running the wrong page's code.

    A stale generation is no longer a trip to the resolver, though. Mac OS X
    moves the generation thousands of times a second, one tlbie per page it
    maps and a segment rewrite per kernel crossing, and each of those used to
    strand every binding in the machine behind a C++ re-resolve. Almost none
    of those translations actually changed. So the exit revalidates in line:
    it probes the primary ITLB for the target address, and an entry fresh
    under the current epoch whose physical page still equals phys means the
    binding is right, the generation is restamped and the jump taken, all
    without leaving generated code. Only a probe miss, a page whose mapping
    really moved, or a genuinely new target reaches the resolver.

    Two ways, because the sites that come through here alternate: a return
    pinging between two callers, the 68k emulator's dispatch bouncing between
    two handlers. Measured, not assumed: with one way, 93% of the misses were
    exactly the address the last rebind had evicted.

    A prediction starts (and is unbound to) 1, an address no instruction can
    have. A binding lives and dies in the way it was installed in and is
    never copied across. resolver remembers the thunk so any unbind knows
    what to put back */
typedef struct ChainVaSlot {
    uint64_t    pred0;
    uint64_t    gen0;
    void*       code0;
    uint64_t    pred1;
    uint64_t    gen1;
    void*       code1;
    const void* resolver;
    uint64_t    flip;   // which way the next eviction lands on
    uint32_t    phys0;  // physical page the way's target resolved to,
    uint32_t    phys1;  // what the inline revalidation checks against
    ChainRef    ref0;   // registry entries owned by the ways; an eviction
    ChainRef    ref1;   // moves the entry, nothing is ever left behind
} ChainVaSlot;

/** rt_chain_resolve for a ChainVaSlot. Installs the target into the way
    already predicting this address if there is one, a virgin way otherwise,
    and the flip way as a last resort, so a site alternating between two
    targets settles with both bound. A megamorphic site just keeps rotating
    its two ways; each rotation moves the way's registry entry, so churn
    costs nothing to anyone else */
const void* rt_chain_resolve_va(ChainVaSlot* slot) noexcept;

/** Hands control back to the emulator between two blocks and says where to
    go next. Everything a native backend used to return to C++ for happens
    here instead, inside the frame the blocks share.

    ppc_state.pc is already the address of the next guest instruction when
    this is called: whichever way the block ended, it wrote it. `retired` is
    how many guest instructions that block completed.

    Returns the code address of the next block, which the caller jumps
    straight to. Returns nullptr when generated code has to stop, which is the
    case for a raised exception, for power going off, for reaching the goal of
    an `until` run, and for a block no emitter took. The caller then unwinds
    its frame and returns, and the loop in ppcjit.cpp sorts out which of those
    it was.

    Defined in ppcjit.cpp rather than here: it is the one runtime entry point
    that needs the lookup and translation side, and putting it there keeps
    find_or_translate private.

    Nothing may unwind out of it. Both the timers it services and the
    instruction fetch it performs can raise, and the frame calling it has no
    unwind information */
const void* rt_dispatch(uint32_t retired) noexcept;

/* There is deliberately no rt_read or rt_write here. A load or store that
   leaves the inline TLB fast path runs the interpreter's helper for the
   whole instruction through rt_call_op, because the instruction has to stay
   atomic: a device read can raise an exception and still complete, and any
   scheme that did only the memory access out of line left rd unwritten
   while the guest, resumed at the same instruction, read the device again */

} // namespace dppc_jit

#endif // PPC_JIT_RUNTIME_H
