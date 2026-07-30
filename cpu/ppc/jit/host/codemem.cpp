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

/** @file Memory that generated code runs from. */

#include "codemem.h"

#include <loguru.hpp>

#if defined(_WIN32)
#  include <windows.h>
#else
#  include <sys/mman.h>
#  include <unistd.h>
#endif

namespace dppc_jit {

namespace {

inline size_t round_up(size_t value, size_t to) {
    return (value + to - 1) & ~(to - 1);
}

size_t host_page_size() {
#if defined(_WIN32)
    SYSTEM_INFO info;
    GetSystemInfo(&info);
    return info.dwPageSize;
#else
    long sz = sysconf(_SC_PAGESIZE);
    return sz > 0 ? size_t(sz) : 4096;
#endif
}

/** Tells the processor that the bytes under it changed.

    x86-64 keeps its caches coherent and needs nothing. AArch64 does not, and
    would happily run whatever was in the instruction cache before */
inline void sync_icache(void* addr, size_t bytes) {
#if defined(DPPC_JIT_AARCH64) && (defined(__GNUC__) || defined(__clang__))
    __builtin___clear_cache(static_cast<char*>(addr), static_cast<char*>(addr) + bytes);
#else
    (void)addr;
    (void)bytes;
#endif
}

} // namespace

CodeMem::~CodeMem() {
    this->shutdown();
}

bool CodeMem::init(size_t bytes, size_t slot_bytes) {
    this->shutdown();

    // the protection flip covers whole pages, so the code part is rounded up
    // and the slot tail starts on the first page the flip leaves alone
    const size_t code_part = round_up(bytes, host_page_size());
    const size_t total     = code_part + round_up(slot_bytes, host_page_size());

#if defined(_WIN32)
    void* mem = VirtualAlloc(nullptr, total, MEM_RESERVE, PAGE_NOACCESS);
    if (!mem) {
        LOG_F(WARNING, "JIT: could not reserve %zu bytes of code memory", total);
        return false;
    }
#else
    // anonymous memory is committed by the kernel one touched page at a
    // time, so reserving generously costs address space and nothing else
    void* mem = mmap(nullptr, total, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mem == MAP_FAILED) {
        LOG_F(WARNING, "JIT: could not reserve %zu bytes of code memory", total);
        return false;
    }
#endif

    this->base        = static_cast<uint8_t*>(mem);
    this->size        = code_part;
    this->offset      = 0;
    this->floor       = 0;
    this->slot_size   = total - code_part;
    this->slot_offset = 0;
    this->reserved    = total;
#if defined(_WIN32)
    this->code_committed = 0;
    this->slot_committed = 0;
    if (!this->ensure_code_committed(host_page_size())) {
        this->shutdown();
        return false;
    }
#else
    this->code_committed = code_part;
    this->slot_committed = total - code_part;
#endif
    this->writable    = true;

    // nothing has been emitted, but leaving it writable would mean the very
    // first end_write is the only thing standing between us and a region that
    // is never made executable
    return this->end_write();
}

void CodeMem::shutdown() {
    if (!this->base) {
        return;
    }
#if defined(_WIN32)
    VirtualFree(this->base, 0, MEM_RELEASE);
#else
    munmap(this->base, this->reserved);
#endif
    this->base        = nullptr;
    this->size        = 0;
    this->offset      = 0;
    this->floor       = 0;
    this->slot_size   = 0;
    this->slot_offset = 0;
    this->reserved    = 0;
    this->code_committed = 0;
    this->slot_committed = 0;
    this->writable    = false;
}

/** Commit granule. Big enough that a storm compiling thousands of blocks a
    second grows the region a handful of times, small enough that an idle
    guest never pays for space it will not use */
constexpr size_t COMMIT_GRANULE = size_t(4) << 20;

bool CodeMem::ensure_code_committed(size_t end) {
#if defined(_WIN32)
    if (end <= this->code_committed) {
        return true;
    }
    const size_t grown = round_up(end, COMMIT_GRANULE) > this->size
                       ? this->size : round_up(end, COMMIT_GRANULE);
    if (!VirtualAlloc(this->base + this->code_committed,
                      grown - this->code_committed, MEM_COMMIT, PAGE_READWRITE)) {
        LOG_F(ERROR, "JIT: could not commit code memory");
        return false;
    }
    this->code_committed = grown;
#else
    (void)end;
#endif
    return true;
}

bool CodeMem::ensure_slot_committed(size_t end) {
#if defined(_WIN32)
    if (end <= this->slot_committed) {
        return true;
    }
    const size_t grown = round_up(end, COMMIT_GRANULE) > this->slot_size
                       ? this->slot_size : round_up(end, COMMIT_GRANULE);
    if (!VirtualAlloc(this->base + this->size + this->slot_committed,
                      grown - this->slot_committed, MEM_COMMIT, PAGE_READWRITE)) {
        LOG_F(ERROR, "JIT: could not commit slot memory");
        return false;
    }
    this->slot_committed = grown;
#else
    (void)end;
#endif
    return true;
}

void CodeMem::reset() {
    // the pages above the floor carry nothing anyone will run again; handing
    // them back is what keeps the footprint at the working set instead of
    // the high water mark of the worst storm so far
    const size_t page       = host_page_size();
    const size_t keep       = round_up(this->floor, page);
    const size_t code_used  = round_up(this->offset, page);
    const size_t slot_used  = round_up(this->slot_offset, page);
    if (this->base && code_used > keep) {
#if defined(_WIN32)
        VirtualFree(this->base + keep, code_used - keep, MEM_DECOMMIT);
        this->code_committed = keep;
#else
        madvise(this->base + keep, code_used - keep, MADV_DONTNEED);
#endif
    }
    if (this->base && slot_used > 0) {
#if defined(_WIN32)
        VirtualFree(this->base + this->size, slot_used, MEM_DECOMMIT);
        this->slot_committed = 0;
#else
        madvise(this->base + this->size, slot_used, MADV_DONTNEED);
#endif
    }
    this->offset      = this->floor;
    this->slot_offset = 0;
}

void CodeMem::mark_floor() {
    this->floor = this->offset;
}

bool CodeMem::begin_write() {
    if (!this->base || this->writable) {
        return this->base != nullptr;
    }
    // the flip stops at the commit watermark: reserved pages have no
    // protection to change yet, and get theirs when they are committed
#if defined(_WIN32)
    DWORD old;
    if (!VirtualProtect(this->base, this->code_committed, PAGE_READWRITE, &old)) {
        LOG_F(ERROR, "JIT: could not make code memory writable");
        return false;
    }
#else
    if (mprotect(this->base, this->code_committed, PROT_READ | PROT_WRITE) != 0) {
        LOG_F(ERROR, "JIT: could not make code memory writable");
        return false;
    }
#endif
    this->writable = true;
    return true;
}

bool CodeMem::end_write() {
    if (!this->base || !this->writable) {
        return this->base != nullptr;
    }
#if defined(_WIN32)
    DWORD old;
    if (!VirtualProtect(this->base, this->code_committed, PAGE_EXECUTE_READ, &old)) {
        LOG_F(ERROR, "JIT: could not make code memory executable");
        return false;
    }
#else
    if (mprotect(this->base, this->code_committed, PROT_READ | PROT_EXEC) != 0) {
        LOG_F(ERROR, "JIT: could not make code memory executable");
        return false;
    }
#endif
    sync_icache(this->base, this->offset);
    this->writable = false;
    return true;
}

bool CodeMem::begin_write_range(size_t upcoming) {
    if (!this->base) {
        return false;
    }
    if (this->writable || this->range_open) {
        return true; // already inside a wider bracket
    }

    // what alloc will touch: from the current offset, aligned up, plus the
    // bytes themselves, all rounded out to whole pages. A region too full to
    // hold it stays untouched and alloc reports the failure
    const size_t page  = host_page_size();
    const size_t start = this->offset & ~(page - 1);
    const size_t end   = round_up(this->offset, 16) + upcoming;
    if (end > this->size) {
        this->range_start = 0;
        this->range_len   = 0;
        this->range_open  = true;
        return true;
    }
    if (!this->ensure_code_committed(round_up(end, page))) {
        return false;
    }
    const size_t len = round_up(end, page) - start;

#if defined(_WIN32)
    DWORD old;
    if (!VirtualProtect(this->base + start, len, PAGE_READWRITE, &old)) {
        LOG_F(ERROR, "JIT: could not make a code range writable");
        return false;
    }
#else
    if (mprotect(this->base + start, len, PROT_READ | PROT_WRITE) != 0) {
        LOG_F(ERROR, "JIT: could not make a code range writable");
        return false;
    }
#endif
    this->range_start = start;
    this->range_len   = len;
    this->range_open  = true;
    return true;
}

bool CodeMem::end_write_range() {
    if (!this->base) {
        return false;
    }
    if (this->writable) {
        return true; // a whole region bracket is open and owns the flip
    }
    if (!this->range_open) {
        return true;
    }

    this->range_open = false;
    if (!this->range_len) {
        return true; // nothing was unprotected
    }

#if defined(_WIN32)
    DWORD old;
    if (!VirtualProtect(this->base + this->range_start, this->range_len,
                        PAGE_EXECUTE_READ, &old)) {
        LOG_F(ERROR, "JIT: could not make a code range executable");
        return false;
    }
#else
    if (mprotect(this->base + this->range_start, this->range_len,
                 PROT_READ | PROT_EXEC) != 0) {
        LOG_F(ERROR, "JIT: could not make a code range executable");
        return false;
    }
#endif
    sync_icache(this->base + this->range_start, this->range_len);
    this->range_len = 0;
    return true;
}

uint8_t* CodeMem::alloc(size_t bytes, size_t alignment) {
    if (!this->base || !(this->writable || this->range_open)) {
        return nullptr;
    }

    const size_t start = round_up(this->offset, alignment);
    if (start + bytes > this->size) {
        return nullptr; // full, the caller declines and the cache gets flushed
    }

    this->offset = start + bytes;
    return this->base + start;
}

uint8_t* CodeMem::slot_alloc(size_t bytes, size_t alignment) {
    if (!this->base) {
        return nullptr;
    }

    const size_t start = round_up(this->slot_offset, alignment);
    if (start + bytes > this->slot_size) {
        return nullptr; // full; the exit is emitted without a chain instead
    }
    if (!this->ensure_slot_committed(start + bytes)) {
        return nullptr;
    }

    this->slot_offset = start + bytes;
    return this->base + this->size + start;
}

} // namespace dppc_jit
