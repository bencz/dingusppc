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

/** @file Memory that generated code runs from.

    This is the one piece of the JIT that has to ask the operating system for
    something, which is why it lives apart from everything else and is only
    built for hosts that have an emitter. A WebAssembly build never compiles
    it, and could not: there is no way to make a page executable there.

    Blocks are handed out by bumping a pointer and are never freed one at a
    time. Individual free would fragment the region for no gain, since a block
    is small and the cache is flushed wholesale often enough. Reclaiming
    happens in reset(), which the code cache calls on a full flush.

    Pages are writable or executable, never both. Emission brackets itself
    with begin_write_range and end_write_range, which flip only the pages the
    block being placed will touch; the whole region flip exists for init.
    Nothing executes while a page is writable because translation and
    execution never overlap in this design.
 */

#ifndef PPC_JIT_CODEMEM_H
#define PPC_JIT_CODEMEM_H

#include <cstddef>
#include <cinttypes>
#include <vector>

namespace dppc_jit {

class CodeMem {
public:
    ~CodeMem();

    /** Reserves the region: `bytes` of code, then `slot_bytes` of plain data
        stuck to its tail. The tail holds the chain slots blocks jump through
        with `jmp qword [rip+disp]`, which is why it has to be this close: a
        rel32 has to reach it from anywhere in the code. It stays writable
        through every protection flip, because binding and unbinding a chain
        happens while code is executing.

        Reserving is not paying: address space is claimed whole so the code
        stays contiguous for those rel32s, but physical memory arrives page
        by page as emission touches it, and reset() hands the pages above
        the floor back to the operating system. The hosts differ in who
        keeps the books: POSIX overcommits anonymous mappings by itself,
        Windows charges commit up front, so there the pages are committed in
        granules as the watermark advances.

        Returns false when the host refuses, which leaves the backend to
        decline every block and everything on the interpreter */
    bool init(size_t bytes, size_t slot_bytes = 0);

    /** Releases the region back to the operating system */
    void shutdown();

    /** Forgets every block without giving the address space back, keeping
        whatever was emitted before the floor was marked. The physical pages
        above the floor go back to the operating system, so the footprint
        after a flush is the stubs, not the storm that forced it */
    void reset();

    /** Declares everything emitted so far permanent, so reset rewinds to here
        instead of to the start. The shared trampoline and dispatch stub live
        below the floor: blocks jump to them by address, so they have to
        outlive every flush */
    void mark_floor();

    /** Makes the region writable. Must bracket every emission */
    bool begin_write();

    /** Makes the region executable again, and syncs the instruction cache on
        architectures that need telling */
    bool end_write();

    /** The per block bracket: flips only the pages the next alloc of
        `upcoming` bytes will touch, instead of the whole region.

        The whole region flip is fine at init, when nothing has been emitted,
        but its cost grows with the region: every page gets its protection
        rewritten and every core its TLB shot down, twice per translation. A
        booting Mac OS X translates thousands of blocks a second, and with a
        64 MB pool the emulator was spending most of its time inside those
        two system calls. A block touches a page or two */
    bool begin_write_range(size_t upcoming);
    bool end_write_range();

    /** Space for one block, aligned, or nullptr when the region is full.
        Only valid between begin_write and end_write */
    uint8_t* alloc(size_t bytes, size_t alignment = 16);

    /** Picks a home for one block: a recycled extent of the right size
        class when one exists, the bump otherwise. Opens the write bracket
        over exactly those pages and returns the spot, nullptr when the
        region is full. Pair with end_write_range */
    uint8_t* alloc_writable(size_t bytes);

    /** Hands a dead block's bytes back for the next compile of its size.
        Only sound at a block boundary, when nothing can be executing them;
        extents below the floor are permanent and stay */
    void recycle(uint8_t* p, size_t bytes);

    /** Space in the data tail, or nullptr when it is full. Usable at any
        time, the tail is never write protected. Nothing here is freed one
        at a time either; reset takes the whole tail back, which is right
        because every slot belongs to some block and the blocks all died */
    uint8_t* slot_alloc(size_t bytes, size_t alignment = 8);

    size_t used() const { return this->offset; }
    size_t capacity() const { return this->size; }

private:
    /** Windows only: makes the pages behind [0, end) of the code part or the
        slot tail real. On POSIX both are committed lazily by the kernel and
        these are no-ops */
    bool ensure_code_committed(size_t end);
    bool ensure_slot_committed(size_t end);

    /** Opens the ranged bracket over the byte span [begin, end), page
        rounded and committed. The begin/end are offsets into the code part */
    bool open_range(size_t begin_off, size_t end_off);

    /** Dead extents by size class, as offsets into the region. Every extent
        is exactly its class in bytes, allocations round up to match, so any
        request of a class fits any extent in its bin */
    static constexpr size_t RECYCLE_CLASS = 256;
    std::vector<std::vector<size_t>> recycle_bins;

    uint8_t* base    = nullptr;
    size_t   size    = 0;      // code bytes only, the tail comes after
    size_t   offset  = 0;
    size_t   floor   = 0;
    size_t   slot_size   = 0;
    size_t   slot_offset = 0;
    size_t   reserved    = 0;  // whole mapping, for the munmap
    size_t   code_committed = 0; // watermarks, only ever below the
    size_t   slot_committed = 0; // reservation on Windows
    bool     writable = false;

    // the open ranged bracket, page aligned; empty when none is open
    size_t   range_start = 0;
    size_t   range_len   = 0;
    bool     range_open  = false;
};

} // namespace dppc_jit

#endif // PPC_JIT_CODEMEM_H
