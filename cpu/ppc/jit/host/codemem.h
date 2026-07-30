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
    with begin_write and end_write, which flip the whole region, so the cost
    is two system calls per translation rather than per block byte. Nothing
    executes while the region is writable because translation and execution
    never overlap in this design.
 */

#ifndef PPC_JIT_CODEMEM_H
#define PPC_JIT_CODEMEM_H

#include <cstddef>
#include <cinttypes>

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

        Returns false when the host refuses, which leaves the backend to
        decline every block and everything on the interpreter */
    bool init(size_t bytes, size_t slot_bytes = 0);

    /** Releases the region back to the operating system */
    void shutdown();

    /** Forgets every block without giving the region back, keeping whatever
        was emitted before the floor was marked */
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

    /** Space for one block, aligned, or nullptr when the region is full.
        Only valid between begin_write and end_write */
    uint8_t* alloc(size_t bytes, size_t alignment = 16);

    /** Space in the data tail, or nullptr when it is full. Usable at any
        time, the tail is never write protected. Nothing here is freed one
        at a time either; reset takes the whole tail back, which is right
        because every slot belongs to some block and the blocks all died */
    uint8_t* slot_alloc(size_t bytes, size_t alignment = 8);

    size_t used() const { return this->offset; }
    size_t capacity() const { return this->size; }

private:
    uint8_t* base    = nullptr;
    size_t   size    = 0;      // code bytes only, the tail comes after
    size_t   offset  = 0;
    size_t   floor   = 0;
    size_t   slot_size   = 0;
    size_t   slot_offset = 0;
    size_t   reserved    = 0;  // whole mapping, for the munmap
    bool     writable = false;
};

} // namespace dppc_jit

#endif // PPC_JIT_CODEMEM_H
