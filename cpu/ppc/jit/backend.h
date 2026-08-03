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

/** @file What a code generator has to provide.

    One implementation per host architecture, plus a portable threaded
    executor retained for explicit differential tests.
 */

#ifndef PPC_JIT_BACKEND_H
#define PPC_JIT_BACKEND_H

#include "jitblock.h"
#include "jitir.h"

#include <memory>

namespace dppc_jit {

class Backend {
public:
    virtual ~Backend() = default;

    virtual const char* name() const = 0;

    /** Turns a decoded block into something runnable.

        Returning nullptr declines the block, which sends its instructions
        back to the interpreter one at a time. That is the escape hatch for
        anything the emitter does not handle yet, so it stays correct while
        coverage grows */
    virtual JitBlock* compile(const IRBlock& ir) = 0;

    /** Hands back one block the code cache dropped.

        Called from the invalidation path, which can fire while the block is
        the one currently executing, so this must not free anything the
        caller could still be walking. ppcjit.cpp defers the free to a block
        boundary for that reason */
    virtual void release(JitBlock* blk) = 0;

    /** Drops whatever the backend still owns after the code cache has
        released every block it knew about, code memory pools and the like */
    virtual void release_all() = 0;

    /** True once the backend has run out of room and is declining blocks it
        would otherwise take.

        Nothing reclaims a single block's code: handing the bytes back one at
        a time would fragment the pool for no gain, and translations are cheap
        to redo. So the pool only ever comes back whole, and this is how the
        backend asks for that to happen. Without it a long run would stay on
        the interpreter once the pool filled and never recover */
    virtual bool wants_flush() const { return false; }
};

/** Portable, executes a block by calling the interpreter helpers in order.
    Selected explicitly for tests; never the automatic fallback */
std::unique_ptr<Backend> make_threaded_backend();

#if defined(DPPC_JIT_X86_64)
std::unique_ptr<Backend> make_x86_64_backend();
#elif defined(DPPC_JIT_AARCH64)
std::unique_ptr<Backend> make_aarch64_backend();
#endif

/** The emitter for this host, or nullptr when there is none */
std::unique_ptr<Backend> make_native_backend();

} // namespace dppc_jit

#endif // PPC_JIT_BACKEND_H
