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

// The floating point opcodes for the processor - ppcfpopcodes.cpp

#include "ppcemu.h"
#include "ppcmacros.h"
#include "ppcmmu.h"
#include <stdlib.h>
#include <cfenv>
#include <cinttypes>
#include <cmath>
#include <cfloat>

inline static void ppc_update_cr1() {
    // copy FPSCR[FX|FEX|VX|OX] to CR1
    ppc_state.cr = (ppc_state.cr & ~CR_select::CR1_field) |
                   ((ppc_state.fpscr >> 4) & CR_select::CR1_field);
}

static int32_t round_to_nearest(double f) {
    return static_cast<int32_t>(std::nearbyint(f));
}

void set_host_rounding_mode(uint8_t mode) {
    switch(mode & FPSCR::RN_MASK) {
    case 0:
        std::fesetround(FE_TONEAREST);
        break;
    case 1:
        std::fesetround(FE_TOWARDZERO);
        break;
    case 2:
        std::fesetround(FE_UPWARD);
        break;
    case 3:
        std::fesetround(FE_DOWNWARD);
        break;
    }
}

void update_fpscr(uint32_t new_fpscr) {
    if ((new_fpscr & FPSCR::RN_MASK) != (ppc_state.fpscr & FPSCR::RN_MASK))
        set_host_rounding_mode(new_fpscr & FPSCR::RN_MASK);

    ppc_state.fpscr = new_fpscr;
}

static int32_t round_to_zero(double f) {
    return static_cast<int32_t>(std::trunc(f));
}

static int32_t round_to_pos_inf(double f) {
    return static_cast<int32_t>(std::ceil(f));
}

static int32_t round_to_neg_inf(double f) {
    return static_cast<int32_t>(std::floor(f));
}

inline static bool check_snan(int check_reg) {
    uint64_t check_int = FPR_INT(check_reg);
    return (((check_int & (0x7FFULL << 52)) == (0x7FFULL << 52)) &&
        ((check_int & ~(0xFFFULL << 52)) != 0ULL) &&
        ((check_int & (0x1ULL << 51)) == 0ULL));
}

inline static bool snan_single_check(int reg_a) {
    if (check_snan(reg_a)) {
        ppc_state.fpscr |= FX | VX | VXSNAN;
        return true;
    }
    return false;
}

inline static bool snan_double_check(int reg_a, int reg_b) {
    if (check_snan(reg_a) || check_snan(reg_b)) {
        ppc_state.fpscr |= FX | VX | VXSNAN;
        return true;
    }
    return false;
}

// ---- IEEE 754 arithmetic core ----------------------------------------------
// FPSCR model per the 60x/750 user manuals: invalid operands raise VX* bits,
// host IEEE flags feed OX/UX/ZX/XX, FR/FI describe the rounding of this
// instruction and FPRF classifies the delivered result. FX fires on any
// 0 to 1 transition of an exception bit, FEX mirrors enabled exceptions.

static constexpr uint32_t FPSCR_VX_BITS =
    FPSCR::VXSNAN | FPSCR::VXISI | FPSCR::VXIDI | FPSCR::VXZDZ | FPSCR::VXIMZ |
    FPSCR::VXVC | FPSCR::VXSOFT | FPSCR::VXSQRT | FPSCR::VXCVI;

static constexpr uint32_t FPSCR_EXC_BITS =
    FPSCR::OX | FPSCR::UX | FPSCR::ZX | FPSCR::XX | FPSCR_VX_BITS;

static void fp_set_exceptions(uint32_t exc_bits) {
    uint32_t fpscr = ppc_state.fpscr;
    if (exc_bits & ~fpscr & FPSCR_EXC_BITS)
        exc_bits |= FPSCR::FX;
    fpscr |= exc_bits;
    if (fpscr & FPSCR_VX_BITS)
        fpscr |= FPSCR::VX;
    else
        fpscr &= ~FPSCR::VX;
    if (((fpscr & FPSCR::VX) && (fpscr & FPSCR::VE)) ||
        ((fpscr & FPSCR::OX) && (fpscr & FPSCR::OE)) ||
        ((fpscr & FPSCR::UX) && (fpscr & FPSCR::UE)) ||
        ((fpscr & FPSCR::ZX) && (fpscr & FPSCR::ZE)) ||
        ((fpscr & FPSCR::XX) && (fpscr & FPSCR::XE)))
        fpscr |= FPSCR::FEX;
    else
        fpscr &= ~FPSCR::FEX;
    ppc_state.fpscr = fpscr;
}

// FPRF field for a result; single selects the single precision denormal range
static uint32_t fp_classify(double r, bool single) {
    if (std::isnan(r))
        return 0x11000;    // QNaN
    if (std::isinf(r))
        return std::signbit(r) ? 0x9000 : 0x5000;
    if (r == 0.0)
        return std::signbit(r) ? 0x12000 : 0x2000;
    if (std::fabs(r) < (single ? double(FLT_MIN) : DBL_MIN))
        return std::signbit(r) ? 0x18000 : 0x14000;    // denormalized
    return std::signbit(r) ? 0x8000 : 0x4000;
}

// NaN and invalid operation results; registers are examined in architectural
// order (frA, frB, frC) and the first NaN found is propagated quieted.
// An enabled invalid operation leaves frD and FPRF untouched.
static void fp_nan_result(int reg_d, uint32_t vx_bits, int r1, int r2, int r3 = -1) {
    uint64_t nan = 0x7FF8000000000000ULL;    // generated QNaN
    const int regs[3] = {r1, r2, r3};
    for (int reg : regs) {
        if (reg >= 0 && std::isnan(GET_FPR(reg))) {
            nan = FPR_INT(reg) | (1ULL << 51);
            break;
        }
    }
    fp_set_exceptions(vx_bits);
    ppc_state.fpscr &= ~(FPSCR::FR | FPSCR::FI);
    if (vx_bits && (ppc_state.fpscr & FPSCR::VE))
        return;
    ppc_state.fpscr = (ppc_state.fpscr & ~FPSCR::FPRF_MASK) | 0x11000;
    ppc_store_fpresult_int(reg_d, nan);
}

// division of a finite nonzero number by zero
static void fp_zx_result(int reg_d, double a, double b) {
    fp_set_exceptions(FPSCR::ZX);
    ppc_state.fpscr &= ~(FPSCR::FR | FPSCR::FI);
    if (ppc_state.fpscr & FPSCR::ZE)
        return;
    double r = std::numeric_limits<double>::infinity();
    if (std::signbit(a) != std::signbit(b))
        r = -r;
    ppc_state.fpscr = (ppc_state.fpscr & ~FPSCR::FPRF_MASK) | fp_classify(r, false);
    ppc_store_fpresult_flt(reg_d, r);
}

// residual of a rounded sum, evaluated with error free transformations in
// round to nearest; the sign of the returned value equals sign((a + b) - s)
static double fp_add_residual(double a, double b, double s) {
    int mode = std::fegetround();
    std::fesetround(FE_TONEAREST);
    double sn  = a + b;
    double bv  = sn - a;
    double av  = sn - bv;
    double err = (a - av) + (b - bv);
    std::fesetround(mode);
    return (sn - s) + err;
}

// residual of a rounded fma, sign true against (a * c + b) - r
static double fp_fma_residual(double a, double c, double b, double r) {
    int mode = std::fegetround();
    std::fesetround(FE_TONEAREST);
    double p  = a * c;
    double ep = std::fma(a, c, -p);
    double s  = p + b;
    double bv = s - p;
    double pv = s - bv;
    double es = (p - pv) + (b - bv);
    std::fesetround(mode);
    return ((s - r) + es) + ep;
}

// FR means the significand was rounded up in magnitude; resid carries the
// sign of (exact - result)
static inline bool fp_rounded_up(double resid, double r) {
    return resid != 0.0 && (std::signbit(resid) != std::signbit(r));
}

// writes a computed result and folds the captured host IEEE flags into the
// FPSCR; fr is the precomputed FR bit for inexact results
static void fp_arith_finish(int reg_d, double r, bool single, int flags, bool fr) {
    uint32_t exc = 0;
    if (flags & FE_OVERFLOW)
        exc |= FPSCR::OX;
    if (flags & FE_UNDERFLOW)
        exc |= FPSCR::UX;
    if (flags & FE_INEXACT)
        exc |= FPSCR::XX;
    fp_set_exceptions(exc);
    uint32_t fpscr = ppc_state.fpscr & ~(FPSCR::FR | FPSCR::FI | FPSCR::FPRF_MASK);
    if (flags & FE_INEXACT) {
        fpscr |= FPSCR::FI;
        if (fr)
            fpscr |= FPSCR::FR;
    }
    ppc_state.fpscr = fpscr | fp_classify(r, single);
    ppc_store_fpresult_flt(reg_d, r);
}

// the 60x/750 multiplier feeds only the top 25 mantissa bits of frC into
// single precision operations, rounding at bit 27
static inline double fp_round_frC(int reg_c) {
    union { uint64_t i; double d; } v;
    v.i = FPR_INT(reg_c);
    v.i = (v.i & 0xFFFFFFFFF8000000ULL) + (v.i & 0x0000000008000000ULL);
    return v.d;
}

static void ppc_update_vx() {
    uint32_t fpscr_check = ppc_state.fpscr & 0x1F80700U;
    if (fpscr_check)
        ppc_state.fpscr |= VX;
    else
        ppc_state.fpscr &= ~VX;
}

static void ppc_update_fex() {
    uint32_t fpscr_check = ((ppc_state.fpscr >> 22) & 0x0F8);
    if (fpscr_check)
        ppc_state.fpscr |= FEX;
    else
        ppc_state.fpscr &= ~FEX;
}

// Floating Point Arithmetic
// shared body of fadd/fsub in both precisions; sub negates frB
template <bool sub, bool single>
static void fp_addsub(int reg_d, int reg_a, int reg_b) {
    double a = GET_FPR(reg_a);
    double b = GET_FPR(reg_b);

    uint32_t vx = 0;
    if (check_snan(reg_a) || check_snan(reg_b))
        vx |= FPSCR::VXSNAN;
    bool any_nan = std::isnan(a) || std::isnan(b);
    if (!any_nan && std::isinf(a) && std::isinf(b) &&
        ((std::signbit(a) != std::signbit(b)) != sub))
        vx |= FPSCR::VXISI;

    if (any_nan || vx) {
        fp_nan_result(reg_d, vx, reg_a, reg_b);
        return;
    }

    double eff_b = sub ? -b : b;
    std::feclearexcept(FE_ALL_EXCEPT);
    double rd = a + eff_b;
    double r  = single ? double(float(rd)) : rd;
    int flags = std::fetestexcept(FE_OVERFLOW | FE_UNDERFLOW | FE_INEXACT);
    bool fr = false;
    if (flags & FE_INEXACT) {
        double as = a, bs = eff_b, rds = rd, rs = r;
        if (flags & FE_OVERFLOW) {
            // FR describes the significand rounding with unbounded exponent,
            // so redo the sum scaled away from the overflow boundary
            int ea, eb;
            std::frexp(a, &ea);
            std::frexp(eff_b, &eb);
            int k = ea > eb ? ea : eb;
            as  = std::scalbn(a, -k);
            bs  = std::scalbn(eff_b, -k);
            rds = as + bs;
            rs  = single ? double(float(rds)) : rds;
        }
        double resid = fp_add_residual(as, bs, rds);
        if (single)
            resid += rds - rs;
        fr = fp_rounded_up(resid, rs);
    }
    fp_arith_finish(reg_d, r, single, flags, fr);
}

template <field_rc rec>
void dppc_interpreter::ppc_fadd(uint32_t opcode) {
    int reg_d = (opcode >> 21) & 31;
    int reg_a = (opcode >> 16) & 31;
    int reg_b = (opcode >> 11) & 31;

    fp_addsub<false, false>(reg_d, reg_a, reg_b);

    if (rec)
        ppc_update_cr1();
}

template void dppc_interpreter::ppc_fadd<RC0>(uint32_t opcode);
template void dppc_interpreter::ppc_fadd<RC1>(uint32_t opcode);

template <field_rc rec>
void dppc_interpreter::ppc_fsub(uint32_t opcode) {
    int reg_d = (opcode >> 21) & 31;
    int reg_a = (opcode >> 16) & 31;
    int reg_b = (opcode >> 11) & 31;

    fp_addsub<true, false>(reg_d, reg_a, reg_b);

    if (rec)
        ppc_update_cr1();
}

template void dppc_interpreter::ppc_fsub<RC0>(uint32_t opcode);
template void dppc_interpreter::ppc_fsub<RC1>(uint32_t opcode);

template <bool single>
static void fp_divide(int reg_d, int reg_a, int reg_b) {
    double a = GET_FPR(reg_a);
    double b = GET_FPR(reg_b);

    if (is_601 && FPR_INT(reg_b) == 0x8000000000000000ULL && a > 0) {
        // the 601 delivers negative zero for positive / -0.0
        fp_arith_finish(reg_d, b, single, 0, false);
        return;
    }

    uint32_t vx = 0;
    if (check_snan(reg_a) || check_snan(reg_b))
        vx |= FPSCR::VXSNAN;
    bool any_nan = std::isnan(a) || std::isnan(b);
    if (!any_nan) {
        if (std::isinf(a) && std::isinf(b))
            vx |= FPSCR::VXIDI;
        if (a == 0.0 && b == 0.0)
            vx |= FPSCR::VXZDZ;
    }

    if (any_nan || vx) {
        fp_nan_result(reg_d, vx, reg_a, reg_b);
        return;
    }

    if (b == 0.0 && !std::isinf(a)) {
        fp_zx_result(reg_d, a, b);
        return;
    }

    std::feclearexcept(FE_ALL_EXCEPT);
    double rd = a / b;
    double r  = single ? double(float(rd)) : rd;
    int flags = std::fetestexcept(FE_OVERFLOW | FE_UNDERFLOW | FE_INEXACT);
    bool fr = false;
    if (flags & FE_INEXACT) {
        double as = a, bs = b, rds = rd, rs = r;
        if (flags & FE_OVERFLOW) {
            int ea, eb;
            std::frexp(a, &ea);
            std::frexp(b, &eb);
            as  = std::scalbn(a, -ea);
            bs  = std::scalbn(b, -eb);
            rds = as / bs;
            rs  = single ? double(float(rds)) : rds;
        }
        double resid = -std::fma(rds, bs, -as) / bs;
        if (single)
            resid += rds - rs;
        fr = fp_rounded_up(resid, rs);
    }
    fp_arith_finish(reg_d, r, single, flags, fr);
}

template <field_rc rec>
void dppc_interpreter::ppc_fdiv(uint32_t opcode) {
    int reg_d = (opcode >> 21) & 31;
    int reg_a = (opcode >> 16) & 31;
    int reg_b = (opcode >> 11) & 31;

    fp_divide<false>(reg_d, reg_a, reg_b);

    if (rec)
        ppc_update_cr1();
}

template void dppc_interpreter::ppc_fdiv<RC0>(uint32_t opcode);
template void dppc_interpreter::ppc_fdiv<RC1>(uint32_t opcode);

template <bool single>
static void fp_multiply(int reg_d, int reg_a, int reg_c) {
    double a = GET_FPR(reg_a);
    double c = GET_FPR(reg_c);

    uint32_t vx = 0;
    if (check_snan(reg_a) || check_snan(reg_c))
        vx |= FPSCR::VXSNAN;
    bool any_nan = std::isnan(a) || std::isnan(c);
    if (!any_nan &&
        ((std::isinf(a) && c == 0.0) || (std::isinf(c) && a == 0.0)))
        vx |= FPSCR::VXIMZ;

    if (any_nan || vx) {
        fp_nan_result(reg_d, vx, reg_a, reg_c);
        return;
    }

    double cm = single ? fp_round_frC(reg_c) : c;
    std::feclearexcept(FE_ALL_EXCEPT);
    double rd = a * cm;
    double r  = single ? double(float(rd)) : rd;
    int flags = std::fetestexcept(FE_OVERFLOW | FE_UNDERFLOW | FE_INEXACT);
    bool fr = false;
    if (flags & FE_INEXACT) {
        double as = a, cs = cm, rds = rd, rs = r;
        if (flags & FE_OVERFLOW) {
            int ea, ec;
            std::frexp(a, &ea);
            std::frexp(cm, &ec);
            as  = std::scalbn(a, -ea);
            cs  = std::scalbn(cm, -ec);
            rds = as * cs;
            rs  = single ? double(float(rds)) : rds;
        }
        double resid = std::fma(as, cs, -rds);
        if (single)
            resid += rds - rs;
        fr = fp_rounded_up(resid, rs);
    }
    fp_arith_finish(reg_d, r, single, flags, fr);
}

template <field_rc rec>
void dppc_interpreter::ppc_fmul(uint32_t opcode) {
    int reg_d = (opcode >> 21) & 31;
    int reg_a = (opcode >> 16) & 31;
    int reg_c = (opcode >> 6) & 31;

    fp_multiply<false>(reg_d, reg_a, reg_c);

    if (rec)
        ppc_update_cr1();
}

template void dppc_interpreter::ppc_fmul<RC0>(uint32_t opcode);
template void dppc_interpreter::ppc_fmul<RC1>(uint32_t opcode);

// shared body of the fmadd family; sub negates frB, neg negates the final
// result unless it is a NaN
template <bool sub, bool neg, bool single>
static void fp_fmadd_family(uint32_t opcode) {
    int reg_d = (opcode >> 21) & 31;
    int reg_a = (opcode >> 16) & 31;
    int reg_b = (opcode >> 11) & 31;
    int reg_c = (opcode >> 6) & 31;
    double a = GET_FPR(reg_a);
    double b = GET_FPR(reg_b);
    double c = GET_FPR(reg_c);

    uint32_t vx = 0;
    if (check_snan(reg_a) || check_snan(reg_b) || check_snan(reg_c))
        vx |= FPSCR::VXSNAN;
    bool nan_ac = std::isnan(a) || std::isnan(c);
    bool imz = !nan_ac &&
        ((std::isinf(a) && c == 0.0) || (std::isinf(c) && a == 0.0));
    if (imz)
        vx |= FPSCR::VXIMZ;
    if (!nan_ac && !std::isnan(b) && !imz &&
        (std::isinf(a) || std::isinf(c)) && std::isinf(b)) {
        bool prod_neg   = std::signbit(a) != std::signbit(c);
        bool addend_neg = std::signbit(b) != sub;
        if (prod_neg != addend_neg)
            vx |= FPSCR::VXISI;
    }

    if (nan_ac || std::isnan(b) || vx) {
        fp_nan_result(reg_d, vx, reg_a, reg_b, reg_c);
        return;
    }

    double cm    = single ? fp_round_frC(reg_c) : c;
    double eff_b = sub ? -b : b;
    std::feclearexcept(FE_ALL_EXCEPT);
    double rd = std::fma(a, cm, eff_b);
    double r  = single ? double(float(rd)) : rd;
    int flags = std::fetestexcept(FE_OVERFLOW | FE_UNDERFLOW | FE_INEXACT);
    bool fr = false;
    if (flags & FE_INEXACT) {
        double as = a, cs = cm, bs = eff_b, rds = rd, rs = r;
        if (flags & FE_OVERFLOW) {
            int ea, ec;
            std::frexp(a, &ea);
            std::frexp(cm, &ec);
            as  = std::scalbn(a, -ea);
            cs  = std::scalbn(cm, -ec);
            bs  = std::scalbn(eff_b, -(ea + ec));
            rds = std::fma(as, cs, bs);
            rs  = single ? double(float(rds)) : rds;
        }
        double resid = fp_fma_residual(as, cs, bs, rds);
        if (single)
            resid += rds - rs;
        fr = fp_rounded_up(resid, rs);
    }
    if (neg)
        r = -r;
    fp_arith_finish(reg_d, r, single, flags, fr);
}

template <field_rc rec>
void dppc_interpreter::ppc_fmadd(uint32_t opcode) {
    fp_fmadd_family<false, false, false>(opcode);

    if (rec)
        ppc_update_cr1();
}

template void dppc_interpreter::ppc_fmadd<RC0>(uint32_t opcode);
template void dppc_interpreter::ppc_fmadd<RC1>(uint32_t opcode);

template <field_rc rec>
void dppc_interpreter::ppc_fmsub(uint32_t opcode) {
    fp_fmadd_family<true, false, false>(opcode);

    if (rec)
        ppc_update_cr1();
}

template void dppc_interpreter::ppc_fmsub<RC0>(uint32_t opcode);
template void dppc_interpreter::ppc_fmsub<RC1>(uint32_t opcode);

template <field_rc rec>
void dppc_interpreter::ppc_fnmadd(uint32_t opcode) {
    fp_fmadd_family<false, true, false>(opcode);

    if (rec)
        ppc_update_cr1();
}

template void dppc_interpreter::ppc_fnmadd<RC0>(uint32_t opcode);
template void dppc_interpreter::ppc_fnmadd<RC1>(uint32_t opcode);

template <field_rc rec>
void dppc_interpreter::ppc_fnmsub(uint32_t opcode) {
    fp_fmadd_family<true, true, false>(opcode);

    if (rec)
        ppc_update_cr1();
}

template void dppc_interpreter::ppc_fnmsub<RC0>(uint32_t opcode);
template void dppc_interpreter::ppc_fnmsub<RC1>(uint32_t opcode);

template <field_rc rec>
void dppc_interpreter::ppc_fadds(uint32_t opcode) {
    int reg_d = (opcode >> 21) & 31;
    int reg_a = (opcode >> 16) & 31;
    int reg_b = (opcode >> 11) & 31;

    fp_addsub<false, true>(reg_d, reg_a, reg_b);

    if (rec)
        ppc_update_cr1();
}

template void dppc_interpreter::ppc_fadds<RC0>(uint32_t opcode);
template void dppc_interpreter::ppc_fadds<RC1>(uint32_t opcode);

template <field_rc rec>
void dppc_interpreter::ppc_fsubs(uint32_t opcode) {
    int reg_d = (opcode >> 21) & 31;
    int reg_a = (opcode >> 16) & 31;
    int reg_b = (opcode >> 11) & 31;

    fp_addsub<true, true>(reg_d, reg_a, reg_b);

    if (rec)
        ppc_update_cr1();
}

template void dppc_interpreter::ppc_fsubs<RC0>(uint32_t opcode);
template void dppc_interpreter::ppc_fsubs<RC1>(uint32_t opcode);

template <field_rc rec>
void dppc_interpreter::ppc_fdivs(uint32_t opcode) {
    int reg_d = (opcode >> 21) & 31;
    int reg_a = (opcode >> 16) & 31;
    int reg_b = (opcode >> 11) & 31;

    fp_divide<true>(reg_d, reg_a, reg_b);

    if (rec)
        ppc_update_cr1();
}

template void dppc_interpreter::ppc_fdivs<RC0>(uint32_t opcode);
template void dppc_interpreter::ppc_fdivs<RC1>(uint32_t opcode);

template <field_rc rec>
void dppc_interpreter::ppc_fmuls(uint32_t opcode) {
    int reg_d = (opcode >> 21) & 31;
    int reg_a = (opcode >> 16) & 31;
    int reg_c = (opcode >> 6) & 31;

    fp_multiply<true>(reg_d, reg_a, reg_c);

    if (rec)
        ppc_update_cr1();
}

template void dppc_interpreter::ppc_fmuls<RC0>(uint32_t opcode);
template void dppc_interpreter::ppc_fmuls<RC1>(uint32_t opcode);

template <field_rc rec>
void dppc_interpreter::ppc_fmadds(uint32_t opcode) {
    fp_fmadd_family<false, false, true>(opcode);

    if (rec)
        ppc_update_cr1();
}

template void dppc_interpreter::ppc_fmadds<RC0>(uint32_t opcode);
template void dppc_interpreter::ppc_fmadds<RC1>(uint32_t opcode);

template <field_rc rec>
void dppc_interpreter::ppc_fmsubs(uint32_t opcode) {
    fp_fmadd_family<true, false, true>(opcode);

    if (rec)
        ppc_update_cr1();
}

template void dppc_interpreter::ppc_fmsubs<RC0>(uint32_t opcode);
template void dppc_interpreter::ppc_fmsubs<RC1>(uint32_t opcode);

template <field_rc rec>
void dppc_interpreter::ppc_fnmadds(uint32_t opcode) {
    fp_fmadd_family<false, true, true>(opcode);

    if (rec)
        ppc_update_cr1();
}

template void dppc_interpreter::ppc_fnmadds<RC0>(uint32_t opcode);
template void dppc_interpreter::ppc_fnmadds<RC1>(uint32_t opcode);

template <field_rc rec>
void dppc_interpreter::ppc_fnmsubs(uint32_t opcode) {
    fp_fmadd_family<true, true, true>(opcode);

    if (rec)
        ppc_update_cr1();
}

template void dppc_interpreter::ppc_fnmsubs<RC0>(uint32_t opcode);
template void dppc_interpreter::ppc_fnmsubs<RC1>(uint32_t opcode);

template <field_rc rec>
void dppc_interpreter::ppc_fabs(uint32_t opcode) {
    ppc_grab_regsfpdb(opcode);

    uint64_t ppc_result64_d = FPR_INT(reg_b) & ~0x8000000000000000U;

    ppc_store_fpresult_int(reg_d, ppc_result64_d);

    if (rec)
        ppc_update_cr1();
}

template void dppc_interpreter::ppc_fabs<RC0>(uint32_t opcode);
template void dppc_interpreter::ppc_fabs<RC1>(uint32_t opcode);

template <field_rc rec>
void dppc_interpreter::ppc_fnabs(uint32_t opcode) {
    ppc_grab_regsfpdb(opcode);

    uint64_t ppc_result64_d = FPR_INT(reg_b) | 0x8000000000000000U;

    ppc_store_fpresult_int(reg_d, ppc_result64_d);

    if (rec)
        ppc_update_cr1();
}

template void dppc_interpreter::ppc_fnabs<RC0>(uint32_t opcode);
template void dppc_interpreter::ppc_fnabs<RC1>(uint32_t opcode);

template <field_rc rec>
void dppc_interpreter::ppc_fneg(uint32_t opcode) {
    ppc_grab_regsfpdb(opcode);

    uint64_t ppc_result64_d = FPR_INT(reg_b) ^ 0x8000000000000000U;

    ppc_store_fpresult_int(reg_d, ppc_result64_d);

    if (rec)
        ppc_update_cr1();
}

template void dppc_interpreter::ppc_fneg<RC0>(uint32_t opcode);
template void dppc_interpreter::ppc_fneg<RC1>(uint32_t opcode);

template <field_rc rec>
void dppc_interpreter::ppc_fsel(uint32_t opcode) {
    ppc_grab_regsfpdabc(opcode);

    double ppc_dblresult64_d = (std::isnan(val_reg_a) || (val_reg_a < 0.0)) ? val_reg_b : val_reg_c;

    ppc_store_fpresult_flt(reg_d, ppc_dblresult64_d);

    if (rec)
        ppc_update_cr1();
}

template void dppc_interpreter::ppc_fsel<RC0>(uint32_t opcode);
template void dppc_interpreter::ppc_fsel<RC1>(uint32_t opcode);

template <field_rc rec>
void dppc_interpreter::ppc_fsqrt(uint32_t opcode) {
    ppc_grab_regsfpdb(opcode);

    double testd2 = (double)(GET_FPR(reg_b));
    double ppc_dblresult64_d = std::sqrt(testd2);

    if (snan_single_check(reg_b)) {
        uint64_t qnan = 0x7FFC000000000000;
        ppc_store_fpresult_int(reg_d, qnan);
    } else {
        ppc_store_fpresult_flt(reg_d, ppc_dblresult64_d);
    }

    if (rec)
        ppc_update_cr1();
}

template void dppc_interpreter::ppc_fsqrt<RC0>(uint32_t opcode);
template void dppc_interpreter::ppc_fsqrt<RC1>(uint32_t opcode);

template <field_rc rec>
void dppc_interpreter::ppc_fsqrts(uint32_t opcode) {
    ppc_grab_regsfpdb(opcode);

    double testd2            = (double)(GET_FPR(reg_b));
    double ppc_dblresult64_d = (float)std::sqrt(testd2);

    if (snan_single_check(reg_b)) {
        uint64_t qnan = 0x7FFC000000000000;
        ppc_store_fpresult_int(reg_d, qnan);
    } else {
        ppc_store_fpresult_flt(reg_d, ppc_dblresult64_d);
    }

    if (rec)
        ppc_update_cr1();
}

template void dppc_interpreter::ppc_fsqrts<RC0>(uint32_t opcode);
template void dppc_interpreter::ppc_fsqrts<RC1>(uint32_t opcode);

template <field_rc rec>
void dppc_interpreter::ppc_frsqrte(uint32_t opcode) {
    ppc_grab_regsfpdb(opcode);

    double testd2            = (double)(GET_FPR(reg_b));
    double ppc_dblresult64_d = 1.0 / sqrt(testd2);

    if (snan_single_check(reg_b)) {
        uint64_t qnan = 0x7FFC000000000000;
        ppc_store_fpresult_int(reg_d, qnan);
    } else {
        ppc_store_fpresult_flt(reg_d, ppc_dblresult64_d);
    }

    if (rec)
        ppc_update_cr1();
}

template void dppc_interpreter::ppc_frsqrte<RC0>(uint32_t opcode);
template void dppc_interpreter::ppc_frsqrte<RC1>(uint32_t opcode);

template <field_rc rec>
void dppc_interpreter::ppc_frsp(uint32_t opcode) {
    ppc_grab_regsfpdb(opcode);

    double ppc_dblresult64_d = (float)(GET_FPR(reg_b));

    if (snan_single_check(reg_b)) {
        uint64_t qnan = 0x7FFC000000000000;
        ppc_store_fpresult_int(reg_d, qnan);
    } else {
        ppc_store_fpresult_flt(reg_d, ppc_dblresult64_d);
    }

    if (rec)
        ppc_update_cr1();
}

template void dppc_interpreter::ppc_frsp<RC0>(uint32_t opcode);
template void dppc_interpreter::ppc_frsp<RC1>(uint32_t opcode);

template <field_rc rec>
void dppc_interpreter::ppc_fres(uint32_t opcode) {
    ppc_grab_regsfpdb(opcode);

    double start_num = GET_FPR(reg_b);
    double ppc_dblresult64_d = (float)(1.0 / start_num);

    if (start_num == 0.0) {
        ppc_state.fpscr |= FPSCR::ZX;
    }
    else if (std::isnan(start_num)) {
        ppc_state.fpscr |= FPSCR::VXSNAN;
    }
    else if (std::isinf(start_num)){
        ppc_state.fpscr &= 0xFFF9FFFF;
        ppc_state.fpscr |= FPSCR::VXSNAN;
    }

    if (snan_single_check(reg_b)) {
        uint64_t qnan = 0x7FFC000000000000;
        ppc_store_fpresult_int(reg_d, qnan);
    } else {
        ppc_store_fpresult_flt(reg_d, ppc_dblresult64_d);
    }

    if (rec)
        ppc_update_cr1();
}

template void dppc_interpreter::ppc_fres<RC0>(uint32_t opcode);
template void dppc_interpreter::ppc_fres<RC1>(uint32_t opcode);

static void round_to_int(uint32_t opcode, const uint8_t mode, field_rc rec) {
    ppc_grab_regsfpdb(opcode);
    double val_reg_b = GET_FPR(reg_b);

    if (std::isnan(val_reg_b)) {
        ppc_state.fpscr &= ~(FPSCR::FR | FPSCR::FI);
        ppc_state.fpscr |= (FPSCR::VXCVI | FPSCR::VX);

        if (check_snan(reg_b)) // issnan
            ppc_state.fpscr |= FPSCR::VXSNAN;

        if (ppc_state.fpscr & FPSCR::VE) {
            ppc_state.fpscr |= FPSCR::FEX; // VX=1 and VE=1 cause FEX to be set
            ppc_floating_point_exception(opcode);
        } else {
            ppc_store_fpresult_int(reg_d, 0xFFF8000080000000ULL);
        }
    } else if (val_reg_b >  static_cast<double>(0x7fffffff) ||
               val_reg_b < -static_cast<double>(0x80000000)) {
        ppc_state.fpscr &= ~(FPSCR::FR | FPSCR::FI);
        ppc_state.fpscr |= (FPSCR::VXCVI | FPSCR::VX);

        if (ppc_state.fpscr & FPSCR::VE) {
            ppc_state.fpscr |= FPSCR::FEX; // VX=1 and VE=1 cause FEX to be set
            ppc_floating_point_exception(opcode);
        } else {
            if (val_reg_b >= 0.0f)
                ppc_store_fpresult_int(reg_d, 0xFFF800007FFFFFFFULL);
            else
                ppc_store_fpresult_int(reg_d, 0xFFF8000080000000ULL);
        }
    } else {
        uint64_t ppc_result64_d;
        switch (mode & 0x3) {
        case 0:
            ppc_result64_d = uint32_t(round_to_nearest(val_reg_b));
            break;
        case 1:
            ppc_result64_d = uint32_t(round_to_zero(val_reg_b));
            break;
        case 2:
            ppc_result64_d = uint32_t(round_to_pos_inf(val_reg_b));
            break;
        case 3:
            ppc_result64_d = uint32_t(round_to_neg_inf(val_reg_b));
            break;
        }

        ppc_result64_d |= 0xFFF8000000000000ULL;

        ppc_store_fpresult_int(reg_d, ppc_result64_d);
    }

    if (rec)
        ppc_update_cr1();
}

template <field_rc rec>
void dppc_interpreter::ppc_fctiw(uint32_t opcode) {
    round_to_int(opcode, ppc_state.fpscr & 0x3, rec);
}

template void dppc_interpreter::ppc_fctiw<RC0>(uint32_t opcode);
template void dppc_interpreter::ppc_fctiw<RC1>(uint32_t opcode);

template <field_rc rec>
void dppc_interpreter::ppc_fctiwz(uint32_t opcode) {
    round_to_int(opcode, 1, rec);
}

template void dppc_interpreter::ppc_fctiwz<RC0>(uint32_t opcode);
template void dppc_interpreter::ppc_fctiwz<RC1>(uint32_t opcode);

// Floating Point Store and Load

void dppc_interpreter::ppc_lfs(uint32_t opcode) {
    ppc_grab_regsfpdia(opcode);
    uint32_t ea = int32_t(int16_t(opcode));
    ea += (reg_a) ? val_reg_a : 0;
    uint32_t result = mmu_read_vmem<uint32_t>(opcode, ea);
    ppc_store_fpresult_flt(reg_d, *(float*)(&result));
}

void dppc_interpreter::ppc_lfsu(uint32_t opcode) {
    ppc_grab_regsfpdia(opcode);

    if (reg_a != 0) {
        uint32_t ea = int32_t(int16_t(opcode));
        ea += val_reg_a;
        uint32_t result = mmu_read_vmem<uint32_t>(opcode, ea);
        ppc_store_fpresult_flt(reg_d, *(float*)(&result));
        ppc_store_iresult_reg(reg_a, ea);
    }
    else {
        ppc_exception_handler(Except_Type::EXC_PROGRAM, Exc_Cause::ILLEGAL_OP);
    }
}

void dppc_interpreter::ppc_lfsx(uint32_t opcode) {
    ppc_grab_regsfpdiab(opcode);
    uint32_t ea = val_reg_b + (reg_a ? val_reg_a : 0);
    uint32_t result = mmu_read_vmem<uint32_t>(opcode, ea);
    ppc_store_fpresult_flt(reg_d, *(float*)(&result));
}

void dppc_interpreter::ppc_lfsux(uint32_t opcode) {
    ppc_grab_regsfpdiab(opcode);

    if (reg_a != 0) {
        uint32_t ea = val_reg_a + val_reg_b;
        uint32_t result = mmu_read_vmem<uint32_t>(opcode, ea);
        ppc_store_fpresult_flt(reg_d, *(float*)(&result));
        ppc_store_iresult_reg(reg_a, ea);
    }
    else {
        ppc_exception_handler(Except_Type::EXC_PROGRAM, Exc_Cause::ILLEGAL_OP);
        return;
    }
}

void dppc_interpreter::ppc_lfd(uint32_t opcode) {
    ppc_grab_regsfpdia(opcode);
    uint32_t ea = int32_t(int16_t(opcode));
    ea += (reg_a) ? val_reg_a : 0;
    uint64_t ppc_result64_d = mmu_read_vmem<uint64_t>(opcode, ea);
    ppc_store_fpresult_int(reg_d, ppc_result64_d);
}

void dppc_interpreter::ppc_lfdu(uint32_t opcode) {
    ppc_grab_regsfpdia(opcode);

    if (reg_a != 0) {
        uint32_t ea = int32_t(int16_t(opcode));
        ea += val_reg_a;
        uint64_t ppc_result64_d = mmu_read_vmem<uint64_t>(opcode, ea);
        ppc_store_fpresult_int(reg_d, ppc_result64_d);
        ppc_store_iresult_reg(reg_a, ea);
    }
    else {
        ppc_exception_handler(Except_Type::EXC_PROGRAM, Exc_Cause::ILLEGAL_OP);
    }
}

void dppc_interpreter::ppc_lfdx(uint32_t opcode) {
    ppc_grab_regsfpdiab(opcode);
    uint32_t ea = val_reg_b + (reg_a ? val_reg_a : 0);
    uint64_t ppc_result64_d = mmu_read_vmem<uint64_t>(opcode, ea);
    ppc_store_fpresult_int(reg_d, ppc_result64_d);
}

void dppc_interpreter::ppc_lfdux(uint32_t opcode) {
    ppc_grab_regsfpdiab(opcode);

    if (reg_a != 0) {
        uint32_t ea = val_reg_a + val_reg_b;
        uint64_t ppc_result64_d = mmu_read_vmem<uint64_t>(opcode, ea);
        ppc_store_fpresult_int(reg_d, ppc_result64_d);
        ppc_store_iresult_reg(reg_a, ea);
    }
    else {
        ppc_exception_handler(Except_Type::EXC_PROGRAM, Exc_Cause::ILLEGAL_OP);
    }
}

void dppc_interpreter::ppc_stfs(uint32_t opcode) {
    ppc_grab_regsfpsia(opcode);
    uint32_t ea = int32_t(int16_t(opcode));
    ea += (reg_a) ? val_reg_a : 0;
    float result = float(GET_FPR(reg_s));
    mmu_write_vmem<uint32_t>(opcode, ea, *(uint32_t*)(&result));
}

void dppc_interpreter::ppc_stfsu(uint32_t opcode) {
    ppc_grab_regsfpsia(opcode);

    if (reg_a != 0) {
        uint32_t ea = int32_t(int16_t(opcode));
        ea += val_reg_a;
        float result = float(GET_FPR(reg_s));
        mmu_write_vmem<uint32_t>(opcode, ea, *(uint32_t*)(&result));
        ppc_store_iresult_reg(reg_a, ea);
    }
    else {
        ppc_exception_handler(Except_Type::EXC_PROGRAM, Exc_Cause::ILLEGAL_OP);
    }
}

void dppc_interpreter::ppc_stfsx(uint32_t opcode) {
    ppc_grab_regsfpsiab(opcode);
    uint32_t ea = val_reg_b + (reg_a ? val_reg_a : 0);
    float result = float(GET_FPR(reg_s));
    mmu_write_vmem<uint32_t>(opcode, ea, *(uint32_t*)(&result));
}

void dppc_interpreter::ppc_stfsux(uint32_t opcode) {
    ppc_grab_regsfpsiab(opcode);

    if (reg_a != 0) {
        uint32_t ea = val_reg_a + val_reg_b;
        float result = float(GET_FPR(reg_s));
        mmu_write_vmem<uint32_t>(opcode, ea, *(uint32_t*)(&result));
        ppc_store_iresult_reg(reg_a, ea);
    }
    else {
        ppc_exception_handler(Except_Type::EXC_PROGRAM, Exc_Cause::ILLEGAL_OP);
    }
}

void dppc_interpreter::ppc_stfd(uint32_t opcode) {
    ppc_grab_regsfpsia(opcode);
    uint32_t ea = int32_t(int16_t(opcode));
    ea += reg_a ? val_reg_a : 0;
    mmu_write_vmem<uint64_t>(opcode, ea, FPR_INT(reg_s));
}

void dppc_interpreter::ppc_stfdu(uint32_t opcode) {
    ppc_grab_regsfpsia(opcode);

    if (reg_a != 0) {
        uint32_t ea = int32_t(int16_t(opcode));
        ea += val_reg_a;
        mmu_write_vmem<uint64_t>(opcode, ea, FPR_INT(reg_s));
        ppc_store_iresult_reg(reg_a, ea);
    }
    else {
        ppc_exception_handler(Except_Type::EXC_PROGRAM, Exc_Cause::ILLEGAL_OP);
    }
}

void dppc_interpreter::ppc_stfdx(uint32_t opcode) {
    ppc_grab_regsfpsiab(opcode);
    uint32_t ea = val_reg_b + (reg_a ? val_reg_a : 0);
    mmu_write_vmem<uint64_t>(opcode, ea, FPR_INT(reg_s));
}

void dppc_interpreter::ppc_stfdux(uint32_t opcode) {
    ppc_grab_regsfpsiab(opcode);

    if (reg_a != 0) {
        uint32_t ea = val_reg_a + val_reg_b;
        mmu_write_vmem<uint64_t>(opcode, ea, FPR_INT(reg_s));
        ppc_store_iresult_reg(reg_a, ea);
    }
    else {
        ppc_exception_handler(Except_Type::EXC_PROGRAM, Exc_Cause::ILLEGAL_OP);
    }
}

void dppc_interpreter::ppc_stfiwx(uint32_t opcode) {
    ppc_grab_regsfpsiab(opcode);
    uint32_t ea = val_reg_b + (reg_a ? val_reg_a : 0);
    mmu_write_vmem<uint32_t>(opcode, ea, uint32_t(FPR_INT(reg_s)));
}

// Floating Point Register Transfer

template <field_rc rec>
void dppc_interpreter::ppc_fmr(uint32_t opcode) {
    ppc_grab_regsfpdb(opcode);
    ppc_store_fpresult_flt(reg_d, GET_FPR(reg_b));

    if (rec)
        ppc_update_cr1();
}

template void dppc_interpreter::ppc_fmr<RC0>(uint32_t opcode);
template void dppc_interpreter::ppc_fmr<RC1>(uint32_t opcode);

template <field_601 for601, field_rc rec>
void dppc_interpreter::ppc_mffs(uint32_t opcode) {
    int reg_d = (opcode >> 21) & 31;

    ppc_store_fpresult_int(reg_d, uint64_t(ppc_state.fpscr) | (for601 ? 0xFFFFFFFF00000000ULL : 0xFFF8000000000000ULL));

    if (rec)
        ppc_update_cr1();
}

template void dppc_interpreter::ppc_mffs<NOT601, RC0>(uint32_t opcode);
template void dppc_interpreter::ppc_mffs<NOT601, RC1>(uint32_t opcode);
template void dppc_interpreter::ppc_mffs<IS601, RC0>(uint32_t opcode);
template void dppc_interpreter::ppc_mffs<IS601, RC1>(uint32_t opcode);

template <field_rc rec>
void dppc_interpreter::ppc_mtfsf(uint32_t opcode) {
    int reg_b  = (opcode >> 11) & 0x1F;
    uint8_t fm = (opcode >> 17) & 0xFF;

    uint32_t cr_mask = 0;

    if (fm == 0xFFU) // the fast case
        cr_mask = 0xFFFFFFFFUL;
    else { // the slow case
        if (fm & 0x80) cr_mask |= 0xF0000000UL;
        if (fm & 0x40) cr_mask |= 0x0F000000UL;
        if (fm & 0x20) cr_mask |= 0x00F00000UL;
        if (fm & 0x10) cr_mask |= 0x000F0000UL;
        if (fm & 0x08) cr_mask |= 0x0000F000UL;
        if (fm & 0x04) cr_mask |= 0x00000F00UL;
        if (fm & 0x02) cr_mask |= 0x000000F0UL;
        if (fm & 0x01) cr_mask |= 0x0000000FUL;
    }

    // ensure neither FEX nor VX will be changed
    cr_mask &= ~(FPSCR::FEX | FPSCR::VX);

    const uint32_t old_fpscr = ppc_state.fpscr;

    // copy FPR[reg_b] to FPSCR under control of cr_mask
    ppc_state.fpscr = (ppc_state.fpscr & ~cr_mask) | (FPR_INT(reg_b) & cr_mask);

    // FEX and VX are derived bits and follow the usual rule
    ppc_update_vx();
    ppc_update_fex();

    // a guest switching rounding modes does it through here
    if ((old_fpscr ^ ppc_state.fpscr) & FPSCR::RN_MASK)
        set_host_rounding_mode(ppc_state.fpscr & FPSCR::RN_MASK);

    if (rec)
        ppc_update_cr1();
}

template void dppc_interpreter::ppc_mtfsf<RC0>(uint32_t opcode);
template void dppc_interpreter::ppc_mtfsf<RC1>(uint32_t opcode);

template <field_rc rec>
void dppc_interpreter::ppc_mtfsfi(uint32_t opcode) {
    int crf_d    = (opcode >> 21) & 0x1C;
    uint32_t imm = (opcode << 16) & 0xF0000000UL;

    // prepare field mask and ensure that neither FEX nor VX will be changed
    uint32_t mask = (0xF0000000UL >> crf_d) & ~(FPSCR::FEX | FPSCR::VX);

    const uint32_t old_fpscr = ppc_state.fpscr;

    // copy imm to FPSCR[crf_d] under control of the field mask
    ppc_state.fpscr = (ppc_state.fpscr & ~mask) | ((imm >> crf_d) & mask);

    // Update FEX and VX according to the "usual rule"
    ppc_update_vx();
    ppc_update_fex();

    if ((old_fpscr ^ ppc_state.fpscr) & FPSCR::RN_MASK)
        set_host_rounding_mode(ppc_state.fpscr & FPSCR::RN_MASK);

    if (rec)
        ppc_update_cr1();
}

template void dppc_interpreter::ppc_mtfsfi<RC0>(uint32_t opcode);
template void dppc_interpreter::ppc_mtfsfi<RC1>(uint32_t opcode);

template <field_rc rec>
void dppc_interpreter::ppc_mtfsb0(uint32_t opcode) {
    int crf_d = (opcode >> 21) & 0x1F;
    if (!crf_d || (crf_d > 2)) { // FEX and VX can't be explicitly cleared
        ppc_state.fpscr &= ~(0x80000000UL >> crf_d);
        ppc_update_vx();
        ppc_update_fex();
        if (crf_d >= 30)
            set_host_rounding_mode(ppc_state.fpscr & FPSCR::RN_MASK);
    }

    if (rec)
        ppc_update_cr1();
}

template void dppc_interpreter::ppc_mtfsb0<RC0>(uint32_t opcode);
template void dppc_interpreter::ppc_mtfsb0<RC1>(uint32_t opcode);

template <field_rc rec>
void dppc_interpreter::ppc_mtfsb1(uint32_t opcode) {
    int crf_d = (opcode >> 21) & 0x1F;
    if (!crf_d || (crf_d > 2)) { // FEX and VX can't be explicitly set
        ppc_state.fpscr |= (0x80000000UL >> crf_d);
        ppc_update_vx();
        ppc_update_fex();
        if (crf_d >= 30)
            set_host_rounding_mode(ppc_state.fpscr & FPSCR::RN_MASK);
    }

    if (rec)
        ppc_update_cr1();
}

template void dppc_interpreter::ppc_mtfsb1<RC0>(uint32_t opcode);
template void dppc_interpreter::ppc_mtfsb1<RC1>(uint32_t opcode);

void dppc_interpreter::ppc_mcrfs(uint32_t opcode) {
    int crf_d = (opcode >> 21) & 0x1C;
    int crf_s = (opcode >> 16) & 0x1C;
    ppc_state.cr = (
        (ppc_state.cr & ~(0xF0000000UL >> crf_d)) |
        (((ppc_state.fpscr << crf_s) & 0xF0000000UL) >> crf_d)
    );
    ppc_state.fpscr &= ~((0xF0000000UL >> crf_s) & (
        // keep only the FPSCR bits that can be explicitly cleared
        FPSCR::FX | FPSCR::OX |
        FPSCR::UX | FPSCR::ZX | FPSCR::XX | FPSCR::VXSNAN |
        FPSCR::VXISI | FPSCR::VXIDI | FPSCR::VXZDZ | FPSCR::VXIMZ |
        FPSCR::VXVC |
        FPSCR::VXSOFT | FPSCR::VXSQRT | FPSCR::VXCVI
    ));
    ppc_update_vx();
    ppc_update_fex();
}

// Floating Point Comparisons

void dppc_interpreter::ppc_fcmpo(uint32_t opcode) {
    ppc_grab_regsfpsab(opcode);

    uint32_t cmp_c = 0;

    if (std::isnan(db_test_a) || std::isnan(db_test_b)) {
        cmp_c |= CRx_bit::CR_SO;
        uint32_t vx;
        if (check_snan(reg_a) || check_snan(reg_b)) {
            vx = FPSCR::VXSNAN;
            // an ordered compare on a quiet operand raises VXVC even for
            // SNaNs when the invalid exception is disabled
            if (!(ppc_state.fpscr & FPSCR::VE))
                vx |= FPSCR::VXVC;
        } else {
            vx = FPSCR::VXVC;
        }
        fp_set_exceptions(vx);
    }
    else if (db_test_a < db_test_b) {
        cmp_c |= CRx_bit::CR_LT;
    }
    else if (db_test_a > db_test_b) {
        cmp_c |= CRx_bit::CR_GT;
    }
    else {
        cmp_c |= CRx_bit::CR_EQ;
    }

    ppc_state.fpscr = (ppc_state.fpscr & ~FPSCR::FPCC_MASK) | (cmp_c >> 16); // update FPCC
    ppc_state.cr = ((ppc_state.cr & ~(0xF0000000 >> crf_d)) | (cmp_c >> crf_d));
}

void dppc_interpreter::ppc_fcmpu(uint32_t opcode) {
    ppc_grab_regsfpsab(opcode);

    uint32_t cmp_c = 0;

    if (std::isnan(db_test_a) || std::isnan(db_test_b)) {
        cmp_c |= CRx_bit::CR_SO;
        if (check_snan(reg_a) || check_snan(reg_b))
            fp_set_exceptions(FPSCR::VXSNAN);
    }
    else if (db_test_a < db_test_b) {
        cmp_c |= CRx_bit::CR_LT;
    }
    else if (db_test_a > db_test_b) {
        cmp_c |= CRx_bit::CR_GT;
    }
    else {
        cmp_c |= CRx_bit::CR_EQ;
    }

    ppc_state.fpscr = (ppc_state.fpscr & ~FPSCR::FPCC_MASK) | (cmp_c >> 16); // update FPCC
    ppc_state.cr    = ((ppc_state.cr & ~(0xF0000000UL >> crf_d)) | (cmp_c >> crf_d));
}
