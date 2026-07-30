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

    A page becomes protected when it receives its first block and stays
    protected while it holds any; the store that finds it empty is what
    releases it, through ppc_code_cache_store_to_page below */
bool ppc_code_cache_page_is_protected(uint32_t phys_addr);

/** A store landed on a protected page. Drops only what it overlapped and
    answers whether the page still holds translated code afterwards; when it
    does not, the page leaves the protected set and the caller may hand
    writability back. While it does, the page must stay routed through the
    slow store path, because healing it while code remains is how a stale
    mapping lets a store slip past the next block unnoticed */
bool ppc_code_cache_store_to_page(uint32_t phys_addr, uint32_t size);

/** Forgets one registration of the handle on the page phys_addr names,
    without running the release callback. The entry is only tombstoned, so
    calling this from inside the release callback is safe even while an
    invalidation is walking the very vector the entry lives in; the next
    walk over that page compacts it away. This is how a block registered on
    two pages, which a walked through cross page call creates, takes its
    other registration with it when either page kills it */
void ppc_code_cache_remove(uint32_t phys_addr, CodeBlockHandle handle);

/** Drops every block overlapping the range and returns how many went away */
unsigned ppc_code_cache_invalidate(uint32_t phys_addr, uint32_t size);

unsigned ppc_code_cache_invalidate_all();

/** Invoked once per dropped block so the generator can release its memory */
void ppc_code_cache_set_release_cb(std::function<void(CodeBlockHandle)> cb);

#endif // PPC_CODE_CACHE_H
