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

/** @file Bookkeeping for translated code blocks.

    Only the invalidation side lives here. Block creation belongs to the code
    generator, which registers each block it produces and gets a callback when
    one is dropped.

    PowerPC leaves coherency between the instruction and data caches to
    software: a guest that writes code has to follow it with icbi before
    executing it. That makes icbi the primary invalidation path here, not a
    safety net bolted onto the store path.

    Blocks are keyed by physical address, because a translation stays valid
    when the virtual mapping moves. Nothing here needs flushing on BAT, SR or
    MSR changes; only the lookup side of the generator has to account for those.

    This is plain C++ with no host specific facility, so it builds under
    Emscripten as well. There is no code generator in a WebAssembly build, so
    the cache simply stays empty and every entry point below early outs.
 */

#ifndef PPC_CODE_CACHE_H
#define PPC_CODE_CACHE_H

#include <cinttypes>
#include <functional>

/** Opaque handle to whatever the code generator produced for a block */
typedef void* CodeBlockHandle;

/** Granularity of icbi on the emulated processors */
constexpr uint32_t PPC_ICACHE_LINE_SIZE = 32;

/** Number of registered blocks. Exposed so the empty check below can inline */
extern unsigned ppc_code_cache_count;

/** Hot paths test this before doing any work. It is always true in builds
    without a code generator, which keeps icbi as cheap as it was before */
inline bool ppc_code_cache_is_empty() {
    return ppc_code_cache_count == 0;
}

inline unsigned ppc_code_cache_num_blocks() {
    return ppc_code_cache_count;
}

/** Drops every block and forgets the release callback */
void ppc_code_cache_init();

/** Drops every block, keeping the release callback in place */
void ppc_code_cache_reset();

/** Registers a block covering [phys_addr, phys_addr + size).

    The block must not cross a page boundary, which matches the execution
    block rule already enforced by ppc_exec_inner */
void ppc_code_cache_add(uint32_t phys_addr, uint32_t size, CodeBlockHandle handle);

/** True when stores to the page holding phys_addr have to take the slow path.
    The MMU asks this when refilling a data TLB entry to decide whether to
    leave PAGE_WRITABLE off.

    A page becomes protected when it receives its first block and stays that
    way until ppc_code_cache_init, including after its last block is dropped.
    Saying so costs a scan of every data TLB entry, and a page holding code
    gains and loses blocks constantly, so letting the answer flap costs far
    more than protecting a page that no longer needs it. Over-protecting only
    ever costs speed: the slow store path does everything the fast one does */
bool ppc_code_cache_page_is_protected(uint32_t phys_addr);

/** Drops every block overlapping the range and returns how many went away */
unsigned ppc_code_cache_invalidate(uint32_t phys_addr, uint32_t size);

unsigned ppc_code_cache_invalidate_all();

/** Invoked once per dropped block so the generator can release its memory */
void ppc_code_cache_set_release_cb(std::function<void(CodeBlockHandle)> cb);

#endif // PPC_CODE_CACHE_H
