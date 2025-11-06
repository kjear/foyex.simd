#ifndef _FOYE_SIMD_FLOATING_HPP_
#define _FOYE_SIMD_FLOATING_HPP_
#pragma once

#include "simd_def.hpp"
#include "simd_utility.hpp"
#include "simd_cvt.hpp"

namespace fyx::simd
{
    struct MXCSR_Guard
    {
        MXCSR_Guard() noexcept : original_mxcsr(_mm_getcsr()) { }
        ~MXCSR_Guard() noexcept { _mm_setcsr(original_mxcsr); }

        MXCSR_Guard(const MXCSR_Guard&) = delete;
        MXCSR_Guard& operator=(const MXCSR_Guard&) = delete;

        MXCSR_Guard(MXCSR_Guard&& other) noexcept
            : original_mxcsr(other.original_mxcsr) 
        {
            other.original_mxcsr = 0;
        }

        bool invalid_operation() const noexcept { return _mm_getcsr() & _MM_EXCEPT_INVALID; }
        bool divide_zero() const noexcept { return _mm_getcsr() & _MM_EXCEPT_DIV_ZERO; }
        bool overflow() const noexcept { return _mm_getcsr() & _MM_EXCEPT_OVERFLOW; }
        bool underflow() const noexcept { return _mm_getcsr() & _MM_EXCEPT_UNDERFLOW; }
        bool inexact() const noexcept { return _mm_getcsr() & _MM_EXCEPT_INEXACT; }
        bool denormal() const noexcept { return _mm_getcsr() & _MM_EXCEPT_DENORM; }

        bool any_exception() const noexcept { return _mm_getcsr() & _MM_EXCEPT_MASK; }

        void clear_exceptions() noexcept 
        {
            unsigned int mxcsr = _mm_getcsr();
            mxcsr &= ~_MM_EXCEPT_MASK;
            _mm_setcsr(mxcsr);
        }

        void set_rounding_mode(unsigned int mode) noexcept 
        {
            unsigned int mxcsr = _mm_getcsr();
            mxcsr &= ~_MM_ROUND_MASK;
            mxcsr |= (mode & _MM_ROUND_MASK);
            _mm_setcsr(mxcsr);
        }

        unsigned int current_rounding_mode() const noexcept 
        {
            return _mm_getcsr() & _MM_ROUND_MASK;
        }

        unsigned int get_original() const noexcept 
        {
            return original_mxcsr;
        }

        unsigned int get_current() const noexcept 
        {
            return _mm_getcsr();
        }

    private:
        unsigned int original_mxcsr;
    };

    mask_from_simd_t<float32x4> not_nan(float32x4 input) { return mask_from_simd_t<float32x4>(_mm_cmpord_ps(input.data, input.data)); }
    mask_from_simd_t<float64x2> not_nan(float64x2 input) { return mask_from_simd_t<float64x2>(_mm_cmpord_pd(input.data, input.data)); }
    mask_from_simd_t<float32x8> not_nan(float32x8 input) { return mask_from_simd_t<float32x8>(_mm256_cmp_ps(input.data, input.data, _CMP_ORD_Q)); }
    mask_from_simd_t<float64x4> not_nan(float64x4 input) { return mask_from_simd_t<float64x4>(_mm256_cmp_pd(input.data, input.data, _CMP_ORD_Q)); }

    mask_from_simd_t<float32x4> is_nan(float32x4 input) { return mask_from_simd_t<float32x4>(_mm_cmpneq_ps(input.data, input.data)); }
    mask_from_simd_t<float64x2> is_nan(float64x2 input) { return mask_from_simd_t<float64x2>(_mm_cmpneq_pd(input.data, input.data)); }
    mask_from_simd_t<float32x8> is_nan(float32x8 input) { return mask_from_simd_t<float32x8>(_mm256_cmp_ps(input.data, input.data, _CMP_NEQ_UQ)); }
    mask_from_simd_t<float64x4> is_nan(float64x4 input) { return mask_from_simd_t<float64x4>(_mm256_cmp_pd(input.data, input.data, _CMP_NEQ_UQ)); }


#if defined(_FOYE_SIMD_HAS_FP16_)
    mask_from_simd_t<float16x8> not_nan(float16x8 input)
    {
        const __m128i vsrc = input.data;
        const __m128i exponent_mask = _mm_set1_epi16(0x7C00);
        const __m128i mantissa_mask = _mm_set1_epi16(0x03FF);
        __m128i exponent = _mm_and_si128(vsrc, exponent_mask);
        __m128i is_inf_nan = _mm_cmpeq_epi16(exponent, exponent_mask);
        __m128i mantissa = _mm_and_si128(vsrc, mantissa_mask);
        __m128i is_nan = _mm_and_si128(is_inf_nan, _mm_cmpgt_epi16(mantissa, _mm_setzero_si128()));
        return mask_from_simd_t<float16x8>{_mm_andnot_si128(is_nan, _mm_set1_epi16(0xFFFF))};
    }
    mask_from_simd_t<float16x16> not_nan(float16x16 input)
    {
        const __m256i vsrc = input.data;
        const __m256i exponent_mask = _mm256_set1_epi16(0x7C00);
        const __m256i mantissa_mask = _mm256_set1_epi16(0x03FF);
        __m256i exponent = _mm256_and_si256(vsrc, exponent_mask);
        __m256i is_inf_nan = _mm256_cmpeq_epi16(exponent, exponent_mask);
        __m256i mantissa = _mm256_and_si256(vsrc, mantissa_mask);
        __m256i is_nan = _mm256_and_si256(is_inf_nan, _mm256_cmpgt_epi16(mantissa, _mm256_setzero_si256()));
        return mask_from_simd_t<float16x16>{_mm256_andnot_si256(is_nan, _mm256_set1_epi16(0xFFFF))};
    }
#endif
#if defined(_FOYE_SIMD_HAS_BF16_)
    mask_from_simd_t<bfloat16x8> not_nan(bfloat16x8 input)
    {
        const __m128i vsrc = input.data;
        const __m128i exp_mask = _mm_set1_epi16(0x7F80);
        const __m128i man_mask = _mm_set1_epi16(0x007F);
        __m128i exp = _mm_and_si128(vsrc, exp_mask);
        __m128i is_exp_max = _mm_cmpeq_epi16(exp, exp_mask);
        __m128i man = _mm_and_si128(vsrc, man_mask);
        __m128i is_man_nonzero = _mm_cmpgt_epi16(man, _mm_setzero_si128());
        return mask_from_simd_t<float16x8>{_mm_and_si128(is_exp_max, is_man_nonzero)};
    }
    mask_from_simd_t<bfloat16x16> not_nan(bfloat16x16 input)
    {
        const __m256i vsrc = input.data;
        const __m256i exp_mask = _mm256_set1_epi16(0x7F80);
        const __m256i man_mask = _mm256_set1_epi16(0x007F);
        __m256i exp = _mm256_and_si256(vsrc, exp_mask);
        __m256i is_exp_max = _mm256_cmpeq_epi16(exp, exp_mask);
        __m256i man = _mm256_and_si256(vsrc, man_mask);
        __m256i is_man_nonzero = _mm256_cmpgt_epi16(man, _mm256_setzero_si256());
        return mask_from_simd_t<bfloat16x16>{_mm256_and_si256(is_exp_max, is_man_nonzero)};
    }
#endif







    mask_from_simd_t<float32x4> not_inf(float32x4 input)
    {
        const __m128i inf_mask = _mm_set1_epi32(0x7F800000);
        __m128i bits = _mm_castps_si128(input.data);

        __m128i abs_bits = _mm_and_si128(bits, _mm_set1_epi32(0x7FFFFFFF));
        __m128i is_inf = _mm_cmpeq_epi32(abs_bits, inf_mask);
        return mask_from_simd_t<float32x4>{ _mm_castsi128_ps(_mm_xor_si128(is_inf, _mm_set1_epi32(0xFFFFFFFF))) };
    }

    mask_from_simd_t<float64x2> not_inf(float64x2 input)
    {
        const __m128i inf_mask = _mm_set1_epi64x(0x7FF0000000000000);
        __m128i bits = _mm_castpd_si128(input.data);
        __m128i abs_bits = _mm_and_si128(bits, _mm_set1_epi64x(0x7FFFFFFFFFFFFFFF));
        __m128i is_inf = _mm_cmpeq_epi64(abs_bits, inf_mask);
        return mask_from_simd_t<float64x2>{ _mm_castsi128_pd(_mm_xor_si128(is_inf, _mm_set1_epi64x(0xFFFFFFFFFFFFFFFF))) };
    }

    mask_from_simd_t<float32x8> not_inf(float32x8 input)
    {
        const __m256i inf_mask = _mm256_set1_epi32(0x7F800000);
        __m256i bits = _mm256_castps_si256(input.data);
        __m256i abs_bits = _mm256_and_si256(bits, _mm256_set1_epi32(0x7FFFFFFF));
        __m256i is_inf = _mm256_cmpeq_epi32(abs_bits, inf_mask);
        return mask_from_simd_t<float32x8>{ _mm256_castsi256_ps(_mm256_xor_si256(is_inf, _mm256_set1_epi32(0xFFFFFFFF))) };
    }

    mask_from_simd_t<float64x4> not_inf(float64x4 input)
    {
        const __m256i inf_mask = _mm256_set1_epi64x(0x7FF0000000000000);
        __m256i bits = _mm256_castpd_si256(input.data);
        __m256i abs_bits = _mm256_and_si256(bits, _mm256_set1_epi64x(0x7FFFFFFFFFFFFFFF));
        __m256i is_inf = _mm256_cmpeq_epi64(abs_bits, inf_mask);
        return mask_from_simd_t<float64x4>{ _mm256_castsi256_pd(_mm256_xor_si256(is_inf, _mm256_set1_epi64x(0xFFFFFFFFFFFFFFFF))) };
    }



    float32x4 interleave_subadd(float32x4 lhs, float32x4 rhs) { return float32x4{ _mm_addsub_ps(lhs.data, rhs.data) }; }
    float32x8 interleave_subadd(float32x8 lhs, float32x8 rhs) { return float32x8{ _mm256_addsub_ps(lhs.data, rhs.data) }; }
    float64x2 interleave_subadd(float64x2 lhs, float64x2 rhs) { return float64x2{ _mm_addsub_pd(lhs.data, rhs.data) }; }
    float64x4 interleave_subadd(float64x4 lhs, float64x4 rhs) { return float64x4{ _mm256_addsub_pd(lhs.data, rhs.data) }; }

#if defined(_FOYE_SIMD_HAS_FP16_)
    float16x8 interleave_subadd(float16x8 lhs, float16x8 rhs)
    {
        return narrowing<float16x8>(
            interleave_subadd(expand<float32x8>(lhs), expand<float32x8>(rhs)));
    }

    float16x16 interleave_subadd(float16x16 lhs, float16x16 rhs)
    {
        float32x8 res_low = interleave_subadd(expand<float32x8>(lhs.low_part()), expand<float32x8>(rhs.low_part()));
        float32x8 res_high = interleave_subadd(expand<float32x8>(lhs.high_part()), expand<float32x8>(rhs.high_part()));
        return merge(narrowing<float16x8>(res_low), narrowing<float16x8>(res_high));
    }
#endif
#if defined(_FOYE_SIMD_HAS_FP16_)
    bfloat16x8 interleave_subadd(bfloat16x8 lhs, bfloat16x8 rhs)
    {
        return narrowing<bfloat16x8>(
            interleave_subadd(expand<float32x8>(lhs), expand<float32x8>(rhs)));
    }

    bfloat16x16 interleave_subadd(bfloat16x16 lhs, bfloat16x16 rhs)
    {
        float32x8 res_low = interleave_subadd(expand<float32x8>(lhs.low_part()), expand<float32x8>(rhs.low_part()));
        float32x8 res_high = interleave_subadd(expand<float32x8>(lhs.high_part()), expand<float32x8>(rhs.high_part()));
        return merge(narrowing<bfloat16x8>(res_low), narrowing<bfloat16x8>(res_high));
    }
#endif


    mask_from_simd_t<float32x4> where_close(float32x4 lhs, float32x4 rhs, float32x4 relative_tolerance, float32x4 absolute_tolerance)
    {
        const __m128 mask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF));
        const __m128 zero = _mm_setzero_ps();
        __m128 diff = _mm_sub_ps(lhs.data, rhs.data);
        __m128 abs_diff = _mm_and_ps(diff, mask);

        __m128 both_zero = _mm_and_ps(
            _mm_cmp_ps(lhs.data, zero, _CMP_EQ_OQ),
            _mm_cmp_ps(rhs.data, zero, _CMP_EQ_OQ)
        );

        __m128 abs_lhs = _mm_and_ps(lhs.data, mask);
        __m128 abs_rhs = _mm_and_ps(rhs.data, mask);
        __m128 max_abs = _mm_max_ps(abs_lhs, abs_rhs);

        __m128 relative_threshold = _mm_mul_ps(relative_tolerance.data, max_abs);
        __m128 total_threshold = _mm_add_ps(absolute_tolerance.data, relative_threshold);

        __m128 is_close = _mm_cmp_ps(abs_diff, total_threshold, _CMP_LE_OQ);
        __m128 result = _mm_or_ps(is_close, both_zero);

        __m128 lhs_finite = _mm_cmp_ps(lhs.data, lhs.data, _CMP_EQ_OQ);
        __m128 rhs_finite = _mm_cmp_ps(rhs.data, rhs.data, _CMP_EQ_OQ);
        __m128 both_finite = _mm_and_ps(lhs_finite, rhs_finite);

        return mask_from_simd_t<float32x4>{ _mm_and_ps(result, both_finite) };
    }

    mask_from_simd_t<float32x8> where_close(float32x8 lhs, float32x8 rhs, float32x8 relative_tolerance, float32x8 absolute_tolerance)
    {
        const __m256 mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
        const __m256 zero = _mm256_setzero_ps();
        __m256 diff = _mm256_sub_ps(lhs.data, rhs.data);
        __m256 abs_diff = _mm256_and_ps(diff, mask);

        __m256 both_zero = _mm256_and_ps(
            _mm256_cmp_ps(lhs.data, zero, _CMP_EQ_OQ),
            _mm256_cmp_ps(rhs.data, zero, _CMP_EQ_OQ)
        );

        __m256 abs_lhs = _mm256_and_ps(lhs.data, mask);
        __m256 abs_rhs = _mm256_and_ps(rhs.data, mask);
        __m256 max_abs = _mm256_max_ps(abs_lhs, abs_rhs);

        __m256 relative_threshold = _mm256_mul_ps(relative_tolerance.data, max_abs);
        __m256 total_threshold = _mm256_add_ps(absolute_tolerance.data, relative_threshold);

        __m256 is_close = _mm256_cmp_ps(abs_diff, total_threshold, _CMP_LE_OQ);

        __m256 result = _mm256_or_ps(is_close, both_zero);

        __m256 lhs_finite = _mm256_cmp_ps(lhs.data, lhs.data, _CMP_EQ_OQ);
        __m256 rhs_finite = _mm256_cmp_ps(rhs.data, rhs.data, _CMP_EQ_OQ);
        __m256 both_finite = _mm256_and_ps(lhs_finite, rhs_finite);

        return mask_from_simd_t<float32x8>{ _mm256_and_ps(result, both_finite) };
    }

    mask_from_simd_t<float64x4> where_close(float64x4 lhs, float64x4 rhs, float64x4 relative_tolerance, float64x4 absolute_tolerance)
    {
        const __m256d mask = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFF));
        const __m256d zero = _mm256_setzero_pd();
        __m256d diff = _mm256_sub_pd(lhs.data, rhs.data);
        __m256d abs_diff = _mm256_and_pd(diff, mask);

        __m256d both_zero = _mm256_and_pd(
            _mm256_cmp_pd(lhs.data, zero, _CMP_EQ_OQ),
            _mm256_cmp_pd(rhs.data, zero, _CMP_EQ_OQ)
        );

        __m256d abs_lhs = _mm256_and_pd(lhs.data, mask);
        __m256d abs_rhs = _mm256_and_pd(rhs.data, mask);
        __m256d max_abs = _mm256_max_pd(abs_lhs, abs_rhs);

        __m256d relative_threshold = _mm256_mul_pd(relative_tolerance.data, max_abs);
        __m256d total_threshold = _mm256_add_pd(absolute_tolerance.data, relative_threshold);

        __m256d is_close = _mm256_cmp_pd(abs_diff, total_threshold, _CMP_LE_OQ);
        __m256d result = _mm256_or_pd(is_close, both_zero);

        __m256d lhs_finite = _mm256_cmp_pd(lhs.data, lhs.data, _CMP_EQ_OQ);
        __m256d rhs_finite = _mm256_cmp_pd(rhs.data, rhs.data, _CMP_EQ_OQ);
        __m256d both_finite = _mm256_and_pd(lhs_finite, rhs_finite);

        return mask_from_simd_t<float64x4>{ _mm256_and_pd(result, both_finite) };
    }

    mask_from_simd_t<float64x2> where_close(float64x2 lhs, float64x2 rhs, float64x2 relative_tolerance, float64x2 absolute_tolerance)
    {
        const __m128d mask = _mm_castsi128_pd(_mm_set1_epi64x(0x7FFFFFFFFFFFFFFF));
        const __m128d zero = _mm_setzero_pd();
        __m128d diff = _mm_sub_pd(lhs.data, rhs.data);
        __m128d abs_diff = _mm_and_pd(diff, mask);

        __m128d both_zero = _mm_and_pd(
            _mm_cmp_pd(lhs.data, zero, _CMP_EQ_OQ),
            _mm_cmp_pd(rhs.data, zero, _CMP_EQ_OQ)
        );

        __m128d lhs_finite = _mm_cmp_pd(lhs.data, lhs.data, _CMP_EQ_OQ);
        __m128d rhs_finite = _mm_cmp_pd(rhs.data, rhs.data, _CMP_EQ_OQ);
        __m128d both_finite = _mm_and_pd(lhs_finite, rhs_finite);

        __m128d abs_lhs = _mm_and_pd(lhs.data, mask);
        __m128d abs_rhs = _mm_and_pd(rhs.data, mask);
        __m128d max_abs = _mm_max_pd(abs_lhs, abs_rhs);

        __m128d threshold = _mm_add_pd(absolute_tolerance.data,
            _mm_mul_pd(relative_tolerance.data, max_abs));

        __m128d is_close = _mm_cmp_pd(abs_diff, threshold, _CMP_LE_OQ);
        __m128d result = _mm_or_pd(is_close, both_zero);

        return mask_from_simd_t<float64x2>{ _mm_and_pd(result, both_finite) };
    }


    namespace detail
    {
        template<typename simd_type>
        simd_type fmod_fp32_soft_simulation(simd_type x, simd_type y)
        {
            const simd_type zero = load_brocast<simd_type>(0.0f);
            const simd_type na_zero = load_brocast<simd_type>(-0.0f);
            const simd_type nan = load_brocast<simd_type>(std::numeric_limits<float>::quiet_NaN());
            const simd_type inf = load_brocast<simd_type>(std::numeric_limits<float>::infinity());

            mask_from_simd_t<simd_type> mask_y_zero = equal(y, zero);

            simd_type abs_x = bitwise_ANDNOT(na_zero, x);
            simd_type abs_y = bitwise_ANDNOT(na_zero, y);

            mask_from_simd_t<simd_type> mask_x_inf = equal(abs_x, inf);
            mask_from_simd_t<simd_type> mask_y_inf = equal(abs_y, inf);

            mask_from_simd_t<simd_type> special_mask{ bitwise_OR(
                mask_y_zero.as_basic_simd<simd_type>(),
                bitwise_AND(
                    mask_x_inf.as_basic_simd<simd_type>(),
                    bitwise_ANDNOT(
                        mask_y_inf.as_basic_simd<simd_type>(),
                        load_brocast<simd_type>(1.0f)))) };

            mask_from_simd_t<simd_type> mask_abs_lt = less(abs_x, abs_y);
            mask_from_simd_t<simd_type> mask_abs_eq = equal(abs_x, abs_y);

            simd_type zero_with_sign = bitwise_AND(x, na_zero);

            simd_type quotient = divide(x, y);
            simd_type integer_part = trunc(quotient);
            simd_type remainder = minus(x, multiplies(integer_part, y));

            remainder = where_assign(remainder, x, mask_abs_lt);
            remainder = where_assign(remainder, zero_with_sign, mask_abs_eq);
            remainder = where_assign(remainder, nan, special_mask);
            return remainder;
        }

        template<typename simd_type>
        simd_type fmod_fp64_soft_simulation(simd_type x, simd_type y)
        {
            const simd_type zero = load_brocast<simd_type>(0.0);
            const simd_type na_zero = load_brocast<simd_type>(-0.0);
            const simd_type nan = load_brocast<simd_type>(std::numeric_limits<double>::quiet_NaN());
            const simd_type inf = load_brocast<simd_type>(std::numeric_limits<double>::infinity());
            const simd_type one = load_brocast<simd_type>(1.0);
            const simd_type max_safe_quotient = load_brocast<simd_type>(1e15);

            mask_from_simd_t<simd_type> mask_y_zero = equal(y, zero);

            simd_type abs_x = bitwise_ANDNOT(na_zero, x);
            simd_type abs_y = bitwise_ANDNOT(na_zero, y);

            simd_type mask_x_inf = bitwise_ANDNOT(abs_x, inf);
            simd_type mask_y_inf = bitwise_ANDNOT(abs_y, inf);

            simd_type special_mask = bitwise_OR(
                mask_y_zero.as_basic_simd<simd_type>(),
                bitwise_AND(mask_x_inf,
                    bitwise_ANDNOT(mask_y_inf, one)));

            mask_from_simd_t<simd_type> mask_abs_lt = less(abs_x, abs_y);
            mask_from_simd_t<simd_type> mask_abs_eq = equal(abs_x, abs_y);
            simd_type zero_with_sign = bitwise_AND(x, na_zero);

            simd_type quotient = divide(x, y);
            
            simd_type abs_quotient = bitwise_ANDNOT(na_zero, quotient);
            mask_from_simd_t<simd_type> need_high_precision = greater(abs_quotient, max_safe_quotient);

            simd_type integer_part_std = trunc(quotient);
            simd_type remainder_std = minus(x, multiplies(integer_part_std, y));

            simd_type integer_part_high = trunc(quotient);

            simd_type remainder_high = minus(x, multiplies(integer_part_high, y));

            simd_type adjustment = load_brocast<simd_type>(0.0);

            simd_type abs_remainder = bitwise_ANDNOT(na_zero, remainder_high);
            mask_from_simd_t<simd_type> need_adjust_up = greater(remainder_high, zero);
            mask_from_simd_t<simd_type> need_adjust_down = less(remainder_high, zero);

            simd_type adjust_up = bitwise_AND(
                need_adjust_up.as_basic_simd<simd_type>(),
                greater_equal(abs_remainder, abs_y).as_basic_simd<simd_type>());

            simd_type adjust_down = bitwise_AND(
                need_adjust_down.as_basic_simd<simd_type>(),
                greater_equal(abs_remainder, abs_y).as_basic_simd<simd_type>());

            adjustment = where_assign(adjustment, y, adjust_up);
            adjustment = where_assign(adjustment, minus(zero, y), adjust_down);

            remainder_high = plus(remainder_high, adjustment);

            simd_type remainder = where_assign(remainder_std, remainder_high, need_high_precision);

            remainder = where_assign(remainder, x, mask_abs_lt);
            remainder = where_assign(remainder, zero_with_sign, mask_abs_eq);
            remainder = where_assign(remainder, nan, special_mask);

            return simd_type{ remainder };
        }
    }

    float32x8 fmod(float32x8 x_, float32x8 y_) { return detail::fmod_fp32_soft_simulation<float32x8>(x_, y_); }
    float32x4 fmod(float32x4 x_, float32x4 y_) { return detail::fmod_fp32_soft_simulation<float32x4>(x_, y_); }
    float64x2 fmod(float64x2 x_, float64x2 y_) { return detail::fmod_fp64_soft_simulation<float64x2>(x_, y_); }
    float64x4 fmod(float64x4 x_, float64x4 y_) { return detail::fmod_fp64_soft_simulation<float64x4>(x_, y_); }
#if defined(_FOYE_SIMD_HAS_FP16_)
    float16x8 fmod(float16x8 a, float16x8 b)
    {
        float32x8 a32{ cvt8lane_fp16_to_fp32(a.data) };
        float32x8 b32{ cvt8lane_fp16_to_fp32(b.data) };
        float32x8 res32 = fyx::simd::fmod(a32, b32);
        return float16x8{ cvt8lane_fp32_to_fp16(res32.data) };
    }
    float16x16 fmod(float16x16 a, float16x16 b)
    {
        float32x8 res32_low = fyx::simd::fmod(
            float32x8{ cvt8lane_fp16_to_fp32(detail::split_low(a.data)) },
            float32x8{ cvt8lane_fp16_to_fp32(detail::split_low(b.data)) });

        float32x8 res32_high = fyx::simd::fmod(
            float32x8{ cvt8lane_fp16_to_fp32(detail::split_high(a.data)) },
            float32x8{ cvt8lane_fp16_to_fp32(detail::split_high(b.data)) });

        return float16x16{ detail::merge(
            cvt8lane_fp32_to_fp16(res32_low.data),
            cvt8lane_fp32_to_fp16(res32_high.data)) };
    }
#endif
#if defined(_FOYE_SIMD_HAS_BF16_)
    bfloat16x8 fmod(bfloat16x8 a, bfloat16x8 b)
    {
        float32x8 a32{ cvt8lane_bf16_to_fp32(a.data) };
        float32x8 b32{ cvt8lane_bf16_to_fp32(b.data) };
        float32x8 res32 = fyx::simd::fmod(a32, b32);
        return bfloat16x8{ cvt8lane_fp32_to_bf16(res32.data) };
    }
    bfloat16x16 fmod(bfloat16x16 a, bfloat16x16 b)
    {
        float32x8 res32_low = fyx::simd::fmod(
            float32x8{ cvt8lane_bf16_to_fp32(detail::split_low(a.data)) },
            float32x8{ cvt8lane_bf16_to_fp32(detail::split_low(b.data)) });

        float32x8 res32_high = fyx::simd::fmod(
            float32x8{ cvt8lane_bf16_to_fp32(detail::split_high(a.data)) },
            float32x8{ cvt8lane_bf16_to_fp32(detail::split_high(b.data)) });

        return bfloat16x16{ detail::merge(
            cvt8lane_fp32_to_bf16(res32_low.data),
            cvt8lane_fp32_to_bf16(res32_high.data)) };
    }
#endif

    float32x8 fma(float32x8 mul_left, float32x8 mul_right, float32x8 add_right) { return float32x8(_mm256_fmadd_ps(mul_left.data, mul_right.data, add_right.data)); }
    float64x4 fma(float64x4 mul_left, float64x4 mul_right, float64x4 add_right) { return float64x4(_mm256_fmadd_pd(mul_left.data, mul_right.data, add_right.data)); }
    float32x4 fma(float32x4 mul_left, float32x4 mul_right, float32x4 add_right) { return float32x4(_mm_fmadd_ps(mul_left.data, mul_right.data, add_right.data)); }
    float64x2 fma(float64x2 mul_left, float64x2 mul_right, float64x2 add_right) { return float64x2(_mm_fmadd_pd(mul_left.data, mul_right.data, add_right.data)); }
#if defined(_FOYE_SIMD_HAS_FP16_)
    float16x8 fma(float16x8 mul_left, float16x8 mul_right, float16x8 add_right)
    {
        return float16x8{ cvt8lane_fp32_to_fp16(
            _mm256_fmadd_ps(cvt8lane_fp16_to_fp32(mul_left.data),
                            cvt8lane_fp16_to_fp32(mul_right.data),
                            cvt8lane_fp16_to_fp32(add_right.data))) };
    }

    float16x16 fma(float16x16 mul_left, float16x16 mul_right, float16x16 add_right)
    {
        return float16x16{ detail::merge(
            cvt8lane_fp32_to_fp16(_mm256_fmadd_ps(
                cvt8lane_fp16_to_fp32(detail::split_low(mul_left.data)),
                cvt8lane_fp16_to_fp32(detail::split_low(mul_right.data)),
                cvt8lane_fp16_to_fp32(detail::split_low(add_right.data)))),
            cvt8lane_fp32_to_fp16(_mm256_fmadd_ps(
                cvt8lane_fp16_to_fp32(detail::split_high(mul_left.data)),
                cvt8lane_fp16_to_fp32(detail::split_high(mul_right.data)),
                cvt8lane_fp16_to_fp32(detail::split_high(add_right.data))))) };
    }
#endif
#if defined(_FOYE_SIMD_HAS_BF16_)
    bfloat16x8 fma(bfloat16x8 mul_left, bfloat16x8 mul_right, bfloat16x8 add_right)
    {
        __m256 vres32 = _mm256_fmadd_ps(cvt8lane_bf16_to_fp32(mul_left.data),
            cvt8lane_bf16_to_fp32(mul_right.data), cvt8lane_bf16_to_fp32(add_right.data));
        return bfloat16x8{ cvt8lane_fp32_to_bf16(vres32) };
    }

    bfloat16x16 fma(bfloat16x16 mul_left, bfloat16x16 mul_right, bfloat16x16 add_right)
    {
        return bfloat16x16{ detail::merge(
            cvt8lane_fp32_to_bf16(_mm256_fmadd_ps(
                cvt8lane_bf16_to_fp32(detail::split_low(mul_left.data)),
                cvt8lane_bf16_to_fp32(detail::split_low(mul_right.data)),
                cvt8lane_bf16_to_fp32(detail::split_low(add_right.data)))),
            cvt8lane_fp32_to_bf16(_mm256_fmadd_ps(
                cvt8lane_bf16_to_fp32(detail::split_high(mul_left.data)),
                cvt8lane_bf16_to_fp32(detail::split_high(mul_right.data)),
                cvt8lane_bf16_to_fp32(detail::split_high(add_right.data))))) };
    }
#endif

    float32x8 fms(float32x8 mul_left, float32x8 mul_right, float32x8 sub_right) { return float32x8(_mm256_fmsub_ps(mul_left.data, mul_right.data, sub_right.data)); }
    float64x4 fms(float64x4 mul_left, float64x4 mul_right, float64x4 sub_right) { return float64x4(_mm256_fmsub_pd(mul_left.data, mul_right.data, sub_right.data)); }
    float32x4 fms(float32x4 mul_left, float32x4 mul_right, float32x4 sub_right) { return float32x4(_mm_fmsub_ps(mul_left.data, mul_right.data, sub_right.data)); }
    float64x2 fms(float64x2 mul_left, float64x2 mul_right, float64x2 sub_right) { return float64x2(_mm_fmsub_pd(mul_left.data, mul_right.data, sub_right.data)); }
#if defined(_FOYE_SIMD_HAS_FP16_)
    float16x8 fms(float16x8 mul_left, float16x8 mul_right, float16x8 sub_right)
    {
        return float16x8{ cvt8lane_fp32_to_fp16(
            _mm256_fmsub_ps(cvt8lane_fp16_to_fp32(mul_left.data),
                            cvt8lane_fp16_to_fp32(mul_right.data),
                            cvt8lane_fp16_to_fp32(sub_right.data))) };
    }

    float16x16 fms(float16x16 mul_left, float16x16 mul_right, float16x16 sub_right)
    {
        return float16x16{ detail::merge(
            cvt8lane_fp32_to_fp16(_mm256_fmsub_ps(
                cvt8lane_fp16_to_fp32(detail::split_low(mul_left.data)),
                cvt8lane_fp16_to_fp32(detail::split_low(mul_right.data)),
                cvt8lane_fp16_to_fp32(detail::split_low(sub_right.data)))),
            cvt8lane_fp32_to_fp16(_mm256_fmsub_ps(
                cvt8lane_fp16_to_fp32(detail::split_high(mul_left.data)),
                cvt8lane_fp16_to_fp32(detail::split_high(mul_right.data)),
                cvt8lane_fp16_to_fp32(detail::split_high(sub_right.data))))) };
    }
#endif
#if defined(_FOYE_SIMD_HAS_BF16_)
    bfloat16x8 fms(bfloat16x8 mul_left, bfloat16x8 mul_right, bfloat16x8 sub_right)
    {
        __m256 vres32 = _mm256_fmsub_ps(cvt8lane_bf16_to_fp32(mul_left.data),
            cvt8lane_bf16_to_fp32(mul_right.data), cvt8lane_bf16_to_fp32(sub_right.data));
        return bfloat16x8{ cvt8lane_fp32_to_bf16(vres32) };
    }

    bfloat16x16 fms(bfloat16x16 mul_left, bfloat16x16 mul_right, bfloat16x16 sub_right)
    {
        return bfloat16x16{ detail::merge(
            cvt8lane_fp32_to_bf16(_mm256_fmsub_ps(
                cvt8lane_bf16_to_fp32(detail::split_low(mul_left.data)),
                cvt8lane_bf16_to_fp32(detail::split_low(mul_right.data)),
                cvt8lane_bf16_to_fp32(detail::split_low(sub_right.data)))),
            cvt8lane_fp32_to_bf16(_mm256_fmsub_ps(
                cvt8lane_bf16_to_fp32(detail::split_high(mul_left.data)),
                cvt8lane_bf16_to_fp32(detail::split_high(mul_right.data)),
                cvt8lane_bf16_to_fp32(detail::split_high(sub_right.data))))) };
    }
#endif

    float32x8 fnma(float32x8 mul_left, float32x8 mul_right, float32x8 add_right) { return float32x8(_mm256_fnmadd_ps(mul_left.data, mul_right.data, add_right.data)); }
    float64x4 fnma(float64x4 mul_left, float64x4 mul_right, float64x4 add_right) { return float64x4(_mm256_fnmadd_pd(mul_left.data, mul_right.data, add_right.data)); }
    float32x4 fnma(float32x4 mul_left, float32x4 mul_right, float32x4 add_right) { return float32x4(_mm_fnmadd_ps(mul_left.data, mul_right.data, add_right.data)); }
    float64x2 fnma(float64x2 mul_left, float64x2 mul_right, float64x2 add_right) { return float64x2(_mm_fnmadd_pd(mul_left.data, mul_right.data, add_right.data)); }
#if defined(_FOYE_SIMD_HAS_FP16_)
    float16x8 fnma(float16x8 mul_left, float16x8 mul_right, float16x8 add_right)
    {
        return float16x8{ cvt8lane_fp32_to_fp16(
            _mm256_fnmadd_ps(cvt8lane_fp16_to_fp32(mul_left.data),
                            cvt8lane_fp16_to_fp32(mul_right.data),
                            cvt8lane_fp16_to_fp32(add_right.data))) };
    }

    float16x16 fnma(float16x16 mul_left, float16x16 mul_right, float16x16 add_right)
    {
        return float16x16{ detail::merge(
            cvt8lane_fp32_to_fp16(_mm256_fnmadd_ps(
                cvt8lane_fp16_to_fp32(detail::split_low(mul_left.data)),
                cvt8lane_fp16_to_fp32(detail::split_low(mul_right.data)),
                cvt8lane_fp16_to_fp32(detail::split_low(add_right.data)))),
            cvt8lane_fp32_to_fp16(_mm256_fnmadd_ps(
                cvt8lane_fp16_to_fp32(detail::split_high(mul_left.data)),
                cvt8lane_fp16_to_fp32(detail::split_high(mul_right.data)),
                cvt8lane_fp16_to_fp32(detail::split_high(add_right.data))))) };
    }
#endif
#if defined(_FOYE_SIMD_HAS_BF16_)
    bfloat16x8 fnma(bfloat16x8 mul_left, bfloat16x8 mul_right, bfloat16x8 add_right)
    {
        __m256 vres32 = _mm256_fnmadd_ps(cvt8lane_bf16_to_fp32(mul_left.data),
            cvt8lane_bf16_to_fp32(mul_right.data), cvt8lane_bf16_to_fp32(add_right.data));
        return bfloat16x8{ cvt8lane_fp32_to_bf16(vres32) };
    }

    bfloat16x16 fnma(bfloat16x16 mul_left, bfloat16x16 mul_right, bfloat16x16 add_right)
    {
        return bfloat16x16{ detail::merge(
            cvt8lane_fp32_to_bf16(_mm256_fnmadd_ps(
                cvt8lane_bf16_to_fp32(detail::split_low(mul_left.data)),
                cvt8lane_bf16_to_fp32(detail::split_low(mul_right.data)),
                cvt8lane_bf16_to_fp32(detail::split_low(add_right.data)))),
            cvt8lane_fp32_to_bf16(_mm256_fnmadd_ps(
                cvt8lane_bf16_to_fp32(detail::split_high(mul_left.data)),
                cvt8lane_bf16_to_fp32(detail::split_high(mul_right.data)),
                cvt8lane_bf16_to_fp32(detail::split_high(add_right.data))))) };
    }
#endif

    float32x8 fnms(float32x8 mul_left, float32x8 mul_right, float32x8 sub_right) { return float32x8(_mm256_fnmsub_ps(mul_left.data, mul_right.data, sub_right.data)); }
    float64x4 fnms(float64x4 mul_left, float64x4 mul_right, float64x4 sub_right) { return float64x4(_mm256_fnmsub_pd(mul_left.data, mul_right.data, sub_right.data)); }
    float32x4 fnms(float32x4 mul_left, float32x4 mul_right, float32x4 sub_right) { return float32x4(_mm_fnmsub_ps(mul_left.data, mul_right.data, sub_right.data)); }
    float64x2 fnms(float64x2 mul_left, float64x2 mul_right, float64x2 sub_right) { return float64x2(_mm_fnmsub_pd(mul_left.data, mul_right.data, sub_right.data)); }
#if defined(_FOYE_SIMD_HAS_FP16_)
    float16x8 fnms(float16x8 mul_left, float16x8 mul_right, float16x8 sub_right)
    {
        return float16x8{ cvt8lane_fp32_to_fp16(
            _mm256_fnmsub_ps(cvt8lane_fp16_to_fp32(mul_left.data),
                            cvt8lane_fp16_to_fp32(mul_right.data),
                            cvt8lane_fp16_to_fp32(sub_right.data))) };
    }

    float16x16 fnms(float16x16 mul_left, float16x16 mul_right, float16x16 sub_right)
    {
        return float16x16{ detail::merge(
            cvt8lane_fp32_to_fp16(_mm256_fnmsub_ps(
                cvt8lane_fp16_to_fp32(detail::split_low(mul_left.data)),
                cvt8lane_fp16_to_fp32(detail::split_low(mul_right.data)),
                cvt8lane_fp16_to_fp32(detail::split_low(sub_right.data)))),
            cvt8lane_fp32_to_fp16(_mm256_fnmsub_ps(
                cvt8lane_fp16_to_fp32(detail::split_high(mul_left.data)),
                cvt8lane_fp16_to_fp32(detail::split_high(mul_right.data)),
                cvt8lane_fp16_to_fp32(detail::split_high(sub_right.data))))) };
    }
#endif
#if defined(_FOYE_SIMD_HAS_BF16_)
    bfloat16x8 fnms(bfloat16x8 mul_left, bfloat16x8 mul_right, bfloat16x8 sub_right)
    {
        __m256 vres32 = _mm256_fnmsub_ps(cvt8lane_bf16_to_fp32(mul_left.data),
            cvt8lane_bf16_to_fp32(mul_right.data), cvt8lane_bf16_to_fp32(sub_right.data));
        return bfloat16x8{ cvt8lane_fp32_to_bf16(vres32) };
    }

    bfloat16x16 fnms(bfloat16x16 mul_left, bfloat16x16 mul_right, bfloat16x16 sub_right)
    {
        return bfloat16x16{ detail::merge(
            cvt8lane_fp32_to_bf16(_mm256_fnmsub_ps(
                cvt8lane_bf16_to_fp32(detail::split_low(mul_left.data)),
                cvt8lane_bf16_to_fp32(detail::split_low(mul_right.data)),
                cvt8lane_bf16_to_fp32(detail::split_low(sub_right.data)))),
            cvt8lane_fp32_to_bf16(_mm256_fnmsub_ps(
                cvt8lane_bf16_to_fp32(detail::split_high(mul_left.data)),
                cvt8lane_bf16_to_fp32(detail::split_high(mul_right.data)),
                cvt8lane_bf16_to_fp32(detail::split_high(sub_right.data))))) };
    }
#endif

    float32x8 fmaddsub(float32x8 mul_left, float32x8 mul_right, float32x8 rhs) { return float32x8{ _mm256_fmaddsub_ps(mul_left.data, mul_right.data, rhs.data) }; }
    float32x4 fmaddsub(float32x4 mul_left, float32x4 mul_right, float32x4 rhs) { return float32x4{ _mm_fmaddsub_ps(mul_left.data, mul_right.data, rhs.data) }; }
    float64x4 fmaddsub(float64x4 mul_left, float64x4 mul_right, float64x4 rhs) { return float64x4{ _mm256_fmaddsub_pd(mul_left.data, mul_right.data, rhs.data) }; }
    float64x2 fmaddsub(float64x2 mul_left, float64x2 mul_right, float64x2 rhs) { return float64x2{ _mm_fmaddsub_pd(mul_left.data, mul_right.data, rhs.data) }; }
#if defined(_FOYE_SIMD_HAS_FP16_)
    float16x8 fmaddsub(float16x8 mul_left, float16x8 mul_right, float16x8 rhs)
    {
        return float16x8{ cvt8lane_fp32_to_fp16(
            _mm256_fmaddsub_ps(cvt8lane_fp16_to_fp32(mul_left.data),
                            cvt8lane_fp16_to_fp32(mul_right.data),
                            cvt8lane_fp16_to_fp32(rhs.data))) };
    }

    float16x16 fmaddsub(float16x16 mul_left, float16x16 mul_right, float16x16 rhs)
    {
        return float16x16{ detail::merge(
            cvt8lane_fp32_to_fp16(_mm256_fmaddsub_ps(
                cvt8lane_fp16_to_fp32(detail::split_low(mul_left.data)),
                cvt8lane_fp16_to_fp32(detail::split_low(mul_right.data)),
                cvt8lane_fp16_to_fp32(detail::split_low(rhs.data)))),
            cvt8lane_fp32_to_fp16(_mm256_fmaddsub_ps(
                cvt8lane_fp16_to_fp32(detail::split_high(mul_left.data)),
                cvt8lane_fp16_to_fp32(detail::split_high(mul_right.data)),
                cvt8lane_fp16_to_fp32(detail::split_high(rhs.data))))) };
    }
#endif
#if defined(_FOYE_SIMD_HAS_BF16_)
    bfloat16x8 fmaddsub(bfloat16x8 mul_left, bfloat16x8 mul_right, bfloat16x8 rhs)
    {
        __m256 vres32 = _mm256_fmaddsub_ps(cvt8lane_bf16_to_fp32(mul_left.data),
            cvt8lane_bf16_to_fp32(mul_right.data), cvt8lane_bf16_to_fp32(rhs.data));
        return bfloat16x8{ cvt8lane_fp32_to_bf16(vres32) };
    }

    bfloat16x16 fmaddsub(bfloat16x16 mul_left, bfloat16x16 mul_right, bfloat16x16 rhs)
    {
        return bfloat16x16{ detail::merge(
            cvt8lane_fp32_to_bf16(_mm256_fmaddsub_ps(
                cvt8lane_bf16_to_fp32(detail::split_low(mul_left.data)),
                cvt8lane_bf16_to_fp32(detail::split_low(mul_right.data)),
                cvt8lane_bf16_to_fp32(detail::split_low(rhs.data)))),
            cvt8lane_fp32_to_bf16(_mm256_fmaddsub_ps(
                cvt8lane_bf16_to_fp32(detail::split_high(mul_left.data)),
                cvt8lane_bf16_to_fp32(detail::split_high(mul_right.data)),
                cvt8lane_bf16_to_fp32(detail::split_high(rhs.data))))) };
    }
#endif


    float32x8 rsqrt(float32x8 input) { return float32x8{ _mm256_rsqrt_ps(input.data) }; }
    float32x4 rsqrt(float32x4 input) { return float32x4{ _mm_rsqrt_ps(input.data) }; }

#if defined(_FOYE_SIMD_HAS_FP16_)
    float16x8 rsqrt(float16x8 input)
    {
        __m256 vres32 = _mm256_rsqrt_ps(cvt8lane_fp16_to_fp32(input.data));
        return float16x8{ cvt8lane_fp32_to_fp16(vres32) };
    }
    float16x16 rsqrt(float16x16 input)
    {
        return float16x16{ detail::merge(
            cvt8lane_fp32_to_fp16(_mm256_rsqrt_ps(cvt8lane_fp16_to_fp32(detail::split_low(input.data)))),
            cvt8lane_fp32_to_fp16(_mm256_rsqrt_ps(cvt8lane_fp16_to_fp32(detail::split_high(input.data))))) };
    }
#endif
#if defined(_FOYE_SIMD_HAS_BF16_)
    bfloat16x8 rsqrt(bfloat16x8 input)
    {
        __m256 vres32 = _mm256_rsqrt_ps(cvt8lane_bf16_to_fp32(input.data));
        return bfloat16x8{ cvt8lane_fp32_to_bf16(vres32) };
    }
    bfloat16x16 rsqrt(bfloat16x16 input)
    {
        return bfloat16x16{ detail::merge(
            cvt8lane_fp32_to_bf16(_mm256_rsqrt_ps(cvt8lane_bf16_to_fp32(detail::split_low(input.data)))),
            cvt8lane_fp32_to_bf16(_mm256_rsqrt_ps(cvt8lane_bf16_to_fp32(detail::split_high(input.data))))) };
    }
#endif

#if defined(_FOYE_SIMD_ENABLE_EMULATED_)
    namespace detail
    {
        template<typename simd_type>
        simd_type rsqrt_soft_simulation(simd_type input)
        {
            constexpr std::size_t N = 2;

            constexpr std::size_t single_width_bits = simd_type::scalar_bit_width;
            using uscalar_type = fyx::simd::detail::integral_t<single_width_bits, false>;
            using usimd_type = basic_simd<uscalar_type, simd_type::bit_width>;

            const usimd_type MAGIC = fyx::simd::load_brocast<usimd_type>(0x5FE6EB50C7B537AA);
            const simd_type HALF = fyx::simd::load_brocast<simd_type>(0.5);
            const simd_type ONE_HALF = fyx::simd::load_brocast<simd_type>(1.5);

            usimd_type i = fyx::simd::reinterpret<usimd_type>(input);
            i = fyx::simd::minus(MAGIC, fyx::simd::shift_right<1>(i));
            simd_type y = fyx::simd::reinterpret<simd_type>(i);

            for (std::size_t iter = 0; iter < N; ++iter)
            {
                simd_type y2 = fyx::simd::multiplies(y, y);
                simd_type half_src = fyx::simd::multiplies(input, HALF);
                simd_type term = fyx::simd::fnma(half_src, y2, ONE_HALF);
                y = fyx::simd::multiplies(y, term);
            }

            return y;
        }
    }

    float64x2 rsqrt(float64x2 input) { return fyx::simd::detail::rsqrt_soft_simulation<float64x2>(input); }
    float64x4 rsqrt(float64x4 input) { return fyx::simd::detail::rsqrt_soft_simulation<float64x4>(input); }
#endif

    float32x8 rcp(float32x8 input) { return float32x8{ _mm256_rcp_ps(input.data) }; }
    float32x4 rcp(float32x4 input) { return float32x4{ _mm_rcp_ps(input.data) }; }
#if defined(_FOYE_SIMD_ENABLE_EMULATED_)
    float64x2 rcp(float64x2 input) { return float64x2{ _mm_div_pd(_mm_set1_pd(1.0), input.data) }; }
    float64x4 rcp(float64x4 input) { return float64x4{ _mm256_div_pd(_mm256_set1_pd(1.0), input.data) }; }
#endif
#if defined(_FOYE_SIMD_HAS_FP16_)
    float16x8 rcp(float16x8 input)
    {
        __m256 vsrc = cvt8lane_fp16_to_fp32(input.data);
        __m256 vres32 = _mm256_rcp_ps(vsrc);
        return float16x8{ cvt8lane_fp32_to_fp16(vres32) };
    }
    float16x16 rcp(float16x16 input)
    {
        __m256 vsrc_low = cvt8lane_fp16_to_fp32(detail::split_low(input.data));
        __m256 vsrc_high = cvt8lane_fp16_to_fp32(detail::split_high(input.data));
        __m128i v_low = cvt8lane_fp32_to_fp16(_mm256_rcp_ps(vsrc_low));
        __m128i v_high = cvt8lane_fp32_to_fp16(_mm256_rcp_ps(vsrc_high));
        return float16x16{ detail::merge(v_low, v_high) };
    }
#endif
#if defined(_FOYE_SIMD_HAS_BF16_)
    bfloat16x8 rcp(bfloat16x8 input)
    {
        __m256 vsrc = cvt8lane_bf16_to_fp32(input.data);
        __m256 vres32 = _mm256_rcp_ps(vsrc);
        return bfloat16x8{ cvt8lane_fp32_to_bf16(vres32) };
    }
    bfloat16x16 rcp(bfloat16x16 input)
    {
        __m256 vsrc_low = cvt8lane_bf16_to_fp32(detail::split_low(input.data));
        __m256 vsrc_high = cvt8lane_bf16_to_fp32(detail::split_high(input.data));
        __m128i v_low = cvt8lane_fp32_to_bf16(_mm256_rcp_ps(vsrc_low));
        __m128i v_high = cvt8lane_fp32_to_bf16(_mm256_rcp_ps(vsrc_high));
        return bfloat16x16{ detail::merge(v_low, v_high) };
    }
#endif

    float32x8 sqrt(float32x8 input) { return float32x8{ _mm256_sqrt_ps(input.data) }; }
    float64x4 sqrt(float64x4 input) { return float64x4{ _mm256_sqrt_pd(input.data) }; }
    float32x4 sqrt(float32x4 input) { return float32x4{ _mm_sqrt_ps(input.data) }; }
    float64x2 sqrt(float64x2 input) { return float64x2{ _mm_sqrt_pd(input.data) }; }
#if defined(_FOYE_SIMD_HAS_FP16_)
    float16x8 sqrt(float16x8 input)
    {
        __m256 vsrc = cvt8lane_fp16_to_fp32(input.data);
        __m256 vres32 = _mm256_sqrt_ps(vsrc);
        return float16x8{ cvt8lane_fp32_to_fp16(vres32) };
    }
    float16x16 sqrt(float16x16 input)
    {
        __m256 vsrc_low = cvt8lane_fp16_to_fp32(detail::split_low(input.data));
        __m256 vsrc_high = cvt8lane_fp16_to_fp32(detail::split_high(input.data));
        __m128i v_low = cvt8lane_fp32_to_fp16(_mm256_sqrt_ps(vsrc_low));
        __m128i v_high = cvt8lane_fp32_to_fp16(_mm256_sqrt_ps(vsrc_high));
        return float16x16{ detail::merge(v_low, v_high) };
    }
#endif
#if defined(_FOYE_SIMD_HAS_BF16_)
    bfloat16x8 sqrt(bfloat16x8 input)
    {
        __m256 vsrc = cvt8lane_bf16_to_fp32(input.data);
        __m256 vres32 = _mm256_sqrt_ps(vsrc);
        return bfloat16x8{ cvt8lane_fp32_to_bf16(vres32) };
    }
    bfloat16x16 sqrt(bfloat16x16 input)
    {
        __m256 vsrc_low = cvt8lane_bf16_to_fp32(detail::split_low(input.data));
        __m256 vsrc_high = cvt8lane_bf16_to_fp32(detail::split_high(input.data));
        __m128i v_low = cvt8lane_fp32_to_bf16(_mm256_sqrt_ps(vsrc_low));
        __m128i v_high = cvt8lane_fp32_to_bf16(_mm256_sqrt_ps(vsrc_high));
        return bfloat16x16{ detail::merge(v_low, v_high) };
    }
#endif

#if defined(FOYE_SIMD_ENABLE_SVML)
    float32x8 exp(float32x8 input) { return float32x8{ _mm256_exp_ps(input.data) }; }
    float64x4 exp(float64x4 input) { return float64x4{ _mm256_exp_pd(input.data) }; }
    float32x4 exp(float32x4 input) { return float32x4{ _mm_exp_ps(input.data) }; }
    float64x2 exp(float64x2 input) { return float64x2{ _mm_exp_pd(input.data) }; }
#else
    namespace detail
    {
        template<typename simd_type>
        simd_type expFP32_soft_simulation(simd_type x)
            requires(fyx::simd::is_basic_simd_v<simd_type> && std::is_same_v<float, typename simd_type::scalar_t>)
        {
            using vsint_t = fyx::simd::basic_simd<fyx::simd::detail::integral_t<
                simd_type::scalar_bit_width, true>, simd_type::bit_width>;

            const simd_type vexp_hi_f32 = fyx::simd::load_brocast<simd_type>(89.f);
            const simd_type vexp_half_fp32 = fyx::simd::load_brocast<simd_type>(0.5f);
            const simd_type vexp_one_fp32 = fyx::simd::load_brocast<simd_type>(1.f);
            const simd_type vexp_LOG2EF_f32 = fyx::simd::load_brocast<simd_type>(1.44269504088896341f);

            const vsint_t exp_bias_s32 = fyx::simd::load_brocast<vsint_t>(0x7f);

            simd_type vexp_x = fyx::simd::max(x, fyx::simd::load_brocast<simd_type>(-88.3762626647949f));
            vexp_x = fyx::simd::min(vexp_x, vexp_hi_f32);

            simd_type vexp = fyx::simd::fma(vexp_x, vexp_LOG2EF_f32, vexp_half_fp32);
            vsint_t vexp_mm = fyx::simd::floor_as_i(vexp);

            vexp = fyx::simd::floating<simd_type>(vexp_mm);
            vexp_mm = fyx::simd::plus(vexp_mm, exp_bias_s32);
            vexp_mm = fyx::simd::shift_left<23>(vexp_mm);

            vexp_x = fyx::simd::fma(vexp, fyx::simd::load_brocast<simd_type>(-6.93359375E-1f), vexp_x);
            vexp_x = fyx::simd::fma(vexp, fyx::simd::load_brocast<simd_type>(2.12194440E-4f), vexp_x);
            simd_type vexp_xx = fyx::simd::multiplies(vexp_x, vexp_x);

            const simd_type vexp_p[6] = {
                fyx::simd::load_brocast<simd_type>(1.9875691500E-4f), 
                fyx::simd::load_brocast<simd_type>(1.3981999507E-3f),
                fyx::simd::load_brocast<simd_type>(8.3334519073E-3f), 
                fyx::simd::load_brocast<simd_type>(4.1665795894E-2f),
                fyx::simd::load_brocast<simd_type>(1.6666665459E-1f), 
                fyx::simd::load_brocast<simd_type>(5.0000001201E-1f) };

            simd_type vexp_y = fyx::simd::fma(vexp_x, vexp_p[0], vexp_p[1]);
            vexp_y = fyx::simd::fma(vexp_y, vexp_x, vexp_p[2]);
            vexp_y = fyx::simd::fma(vexp_y, vexp_x, vexp_p[3]);
            vexp_y = fyx::simd::fma(vexp_y, vexp_x, vexp_p[4]);
            vexp_y = fyx::simd::fma(vexp_y, vexp_x, vexp_p[5]);

            vexp_y = fyx::simd::fma(vexp_y, vexp_xx, vexp_x);
            vexp_y = fyx::simd::plus(vexp_y, vexp_one_fp32);
            vexp_y = fyx::simd::multiplies(vexp_y, fyx::simd::reinterpret<simd_type>(vexp_mm));

            fyx::simd::mask_from_simd_t<simd_type> mask_not_nan = fyx::simd::not_nan(x);
            simd_type all_Nah = fyx::simd::reinterpret<simd_type>(fyx::simd::load_brocast<vsint_t>(0x7fc00000));
            return fyx::simd::where_assign<simd_type>(all_Nah, vexp_y, mask_not_nan);
        }

        template<typename simd_type>
        simd_type expFP64_soft_simulation(simd_type input)
            requires(fyx::simd::is_basic_simd_v<simd_type>&& std::is_same_v<double, typename simd_type::scalar_t>)
        {
            using int_vector_t = fyx::simd::basic_simd<fyx::simd::detail::integral_t<
                simd_type::scalar_bit_width, true>, simd_type::bit_width>;

            const int_vector_t exponent_bias = fyx::simd::load_brocast<int_vector_t>(0x3ff);

            simd_type clamped_input = fyx::simd::max(input, 
                fyx::simd::load_brocast<simd_type>(-709.43613930310391424428));

            clamped_input = fyx::simd::min(clamped_input, 
                fyx::simd::load_brocast<simd_type>(710.));

            simd_type result = fyx::simd::fma(clamped_input,
                fyx::simd::load_brocast<simd_type>(1.44269504088896340736),
                fyx::simd::load_brocast<simd_type>(0.5));

            int_vector_t exponent_int = fyx::simd::floor_as_i(result);

            result = fyx::simd::floating<simd_type>(exponent_int);
            exponent_int = fyx::simd::plus(exponent_int, exponent_bias);
            exponent_int = fyx::simd::shift_left<52>(exponent_int);

            simd_type fractional_part = fyx::simd::fma(
                result, 
                fyx::simd::load_brocast<simd_type>(-6.93145751953125E-1), 
                clamped_input);

            fractional_part = fyx::simd::fma(
                result, 
                fyx::simd::load_brocast<simd_type>(-1.42860682030941723212E-6), 
                fractional_part);

            simd_type fractional_squared = fyx::simd::multiplies(fractional_part, fractional_part);

            const simd_type numerator_coeff0 = fyx::simd::load_brocast<simd_type>(1.26177193074810590878E-4);
            const simd_type numerator_coeff1 = fyx::simd::load_brocast<simd_type>(3.02994407707441961300E-2);
            const simd_type numerator_coeff2 = fyx::simd::load_brocast<simd_type>(9.99999999999999999910E-1);
            simd_type numerator = fyx::simd::fma(fractional_squared, numerator_coeff0, numerator_coeff1);
            numerator = fyx::simd::fma(numerator, fractional_squared, numerator_coeff2);
            numerator = fyx::simd::multiplies(numerator, fractional_part);

            const simd_type denominator_coeff0 = fyx::simd::load_brocast<simd_type>(3.00198505138664455042E-6);
            const simd_type denominator_coeff1 = fyx::simd::load_brocast<simd_type>(2.52448340349684104192E-3);
            const simd_type denominator_coeff2 = fyx::simd::load_brocast<simd_type>(2.27265548208155028766E-1);
            const simd_type denominator_coeff3 = fyx::simd::load_brocast<simd_type>(2.00000000000000000009E0);
            simd_type denominator = fyx::simd::fma(fractional_squared, denominator_coeff0, denominator_coeff1);
            denominator = fyx::simd::fma(fractional_squared, denominator, denominator_coeff2);
            denominator = fyx::simd::fma(fractional_squared, denominator, denominator_coeff3);

            denominator = fyx::simd::divide(numerator, fyx::simd::minus(denominator, numerator));
            denominator = fyx::simd::fma(fyx::simd::load_brocast<simd_type>(2.0), denominator, fyx::simd::load_brocast<simd_type>(1.0));

            denominator = fyx::simd::multiplies(denominator, fyx::simd::reinterpret<simd_type>(exponent_int));

            fyx::simd::mask_from_simd_t<simd_type> valid_input_mask = fyx::simd::not_nan(input);
            return simd_type{ fyx::simd::where_assign(input, denominator, valid_input_mask) };
        }
    }

    float32x8 exp(float32x8 input) { return float32x8{ fyx::simd::detail::expFP32_soft_simulation<float32x8>(input) }; }
    float64x4 exp(float64x4 input) { return float64x4{ fyx::simd::detail::expFP64_soft_simulation<float64x4>(input) }; }
    float32x4 exp(float32x4 input) { return float32x4{ fyx::simd::detail::expFP32_soft_simulation<float32x4>(input) }; }
    float64x2 exp(float64x2 input) { return float64x2{ fyx::simd::detail::expFP64_soft_simulation<float64x2>(input) }; }
#if defined(_FOYE_SIMD_HAS_FP16_)
    float16x8 exp(float16x8 input)
    {
        float32x8 a32{ cvt8lane_fp16_to_fp32(input.data) };
        float32x8 res32 = fyx::simd::exp(a32);
        return float16x8{ cvt8lane_fp32_to_fp16(res32.data) };
    }
    float16x16 exp(float16x16 input)
    {
        float32x8 res32_low = fyx::simd::exp(float32x8{ cvt8lane_fp16_to_fp32(detail::split_low(input.data)) });
        float32x8 res32_high = fyx::simd::exp(float32x8{ cvt8lane_fp16_to_fp32(detail::split_high(input.data)) });

        return float16x16{ detail::merge(
            cvt8lane_fp32_to_fp16(res32_low.data),
            cvt8lane_fp32_to_fp16(res32_high.data)) };
    }
#endif
#if defined(_FOYE_SIMD_HAS_BF16_)
    bfloat16x8 exp(bfloat16x8 input)
    {
        float32x8 a32{ cvt8lane_bf16_to_fp32(input.data) };
        float32x8 res32 = fyx::simd::exp(a32);
        return bfloat16x8{ cvt8lane_fp32_to_bf16(res32.data) };
    }
    bfloat16x16 exp(bfloat16x16 input)
    {
        float32x8 res32_low = fyx::simd::exp(float32x8{ cvt8lane_bf16_to_fp32(detail::split_low(input.data)) });
        float32x8 res32_high = fyx::simd::exp(float32x8{ cvt8lane_bf16_to_fp32(detail::split_high(input.data)) });

        return bfloat16x16{ detail::merge(
            cvt8lane_fp32_to_bf16(res32_low.data),
            cvt8lane_fp32_to_bf16(res32_high.data)) };
    }
#endif
#endif

#if defined(FOYE_SIMD_ENABLE_SVML)
    float32x8 hypot(float32x8 a, float32x8 b) { return float32x8{ _mm256_hypot_ps(a.data, b.data) }; }
    float64x4 hypot(float64x4 a, float64x4 b) { return float64x4{ _mm256_hypot_pd(a.data, b.data) }; }
    float32x4 hypot(float32x4 a, float32x4 b) { return float32x4{ _mm_hypot_ps(a.data, b.data) }; }
    float64x2 hypot(float64x2 a, float64x2 b) { return float64x2{ _mm_hypot_pd(a.data, b.data) }; }
#else
    namespace detail
    {
        template<typename simd_type>
        simd_type hypot_soft_simulation(simd_type a, simd_type b)
        {
            const simd_type na_zero = fyx::simd::load_brocast<simd_type>(-0.0f);
            simd_type abs_a = fyx::simd::bitwise_ANDNOT(na_zero, a);
            simd_type abs_b = fyx::simd::bitwise_ANDNOT(na_zero, b);

            simd_type max_val = fyx::simd::max(abs_a, abs_b);
            simd_type min_val = fyx::simd::min(abs_a, abs_b);

            simd_type zero = fyx::simd::allzero_bits_as<simd_type>();
            simd_type min_nonzero = fyx::simd::max(min_val, fyx::simd::load_brocast<simd_type>(1e-38f));

            simd_type ratio = fyx::simd::divide(min_nonzero, max_val);

            simd_type ratio2 = fyx::simd::multiplies(ratio, ratio);
            simd_type sqrt_val = fyx::simd::sqrt(fyx::simd::plus(fyx::simd::load_brocast<simd_type>(1.0f), ratio2));
            return fyx::simd::multiplies(max_val, sqrt_val);
        }
    }

    float32x8 hypot(float32x8 a, float32x8 b) { return fyx::simd::detail::hypot_soft_simulation<float32x8>(a, b); }
    float64x4 hypot(float64x4 a, float64x4 b) { return fyx::simd::detail::hypot_soft_simulation<float64x4>(a, b); }
    float32x4 hypot(float32x4 a, float32x4 b) { return fyx::simd::detail::hypot_soft_simulation<float32x4>(a, b); }
    float64x2 hypot(float64x2 a, float64x2 b) { return fyx::simd::detail::hypot_soft_simulation<float64x2>(a, b); }
#endif
#if defined(_FOYE_SIMD_HAS_FP16_)
    float16x8 hypot(float16x8 a, float16x8 b)
    {
        float32x8 a32{ cvt8lane_fp16_to_fp32(a.data) };
        float32x8 b32{ cvt8lane_fp16_to_fp32(b.data) };
        float32x8 res32 = fyx::simd::hypot(a32, b32);
        return float16x8{ cvt8lane_fp32_to_fp16(res32.data) };
    }
    float16x16 hypot(float16x16 a, float16x16 b)
    {
        float32x8 res32_low = fyx::simd::hypot(
            float32x8{ cvt8lane_fp16_to_fp32(detail::split_low(a.data)) },
            float32x8{ cvt8lane_fp16_to_fp32(detail::split_low(b.data)) });

        float32x8 res32_high = fyx::simd::hypot(
            float32x8{ cvt8lane_fp16_to_fp32(detail::split_high(a.data)) },
            float32x8{ cvt8lane_fp16_to_fp32(detail::split_high(b.data)) });

        return float16x16{ detail::merge(
            cvt8lane_fp32_to_fp16(res32_low.data),
            cvt8lane_fp32_to_fp16(res32_high.data)) };
    }
#endif
#if defined(_FOYE_SIMD_HAS_BF16_)
    bfloat16x8 hypot(bfloat16x8 a, bfloat16x8 b)
    {
        float32x8 a32{ cvt8lane_bf16_to_fp32(a.data) };
        float32x8 b32{ cvt8lane_bf16_to_fp32(b.data) };
        float32x8 res32 = fyx::simd::hypot(a32, b32);
        return bfloat16x8{ cvt8lane_fp32_to_bf16(res32.data) };
    }
    bfloat16x16 hypot(bfloat16x16 a, bfloat16x16 b)
    {
        float32x8 res32_low = fyx::simd::hypot(
            float32x8{ cvt8lane_bf16_to_fp32(detail::split_low(a.data)) },
            float32x8{ cvt8lane_bf16_to_fp32(detail::split_low(b.data)) });

        float32x8 res32_high = fyx::simd::hypot(
            float32x8{ cvt8lane_bf16_to_fp32(detail::split_high(a.data)) },
            float32x8{ cvt8lane_bf16_to_fp32(detail::split_high(b.data)) });

        return bfloat16x16{ detail::merge(
            cvt8lane_fp32_to_bf16(res32_low.data),
            cvt8lane_fp32_to_bf16(res32_high.data)) };
    }
#endif

#if defined(FOYE_SIMD_ENABLE_SVML)
    float32x8 erf(float32x8 input) { return float32x8{ _mm256_erf_ps(input.data) }; }
    float64x4 erf(float64x4 input) { return float64x4{ _mm256_erf_pd(input.data) }; }
    float32x4 erf(float32x4 input) { return float32x4{ _mm_erf_ps(input.data) }; }
    float64x2 erf(float64x2 input) { return float64x2{ _mm_erf_pd(input.data) }; }
#else
    namespace detail
    {
        template<typename simd_type>
        simd_type erf_soft_simulation(simd_type v)
        {
            const simd_type coef0 = fyx::simd::load_brocast<simd_type>(0.3275911);
            const simd_type coef1 = fyx::simd::load_brocast<simd_type>(1.061405429);
            const simd_type coef2 = fyx::simd::load_brocast<simd_type>(-1.453152027);
            const simd_type coef3 = fyx::simd::load_brocast<simd_type>(1.421413741);
            const simd_type coef4 = fyx::simd::load_brocast<simd_type>(-0.284496736);
            const simd_type coef5 = fyx::simd::load_brocast<simd_type>(0.254829592);
            const simd_type ones = fyx::simd::load_brocast<simd_type>(1.0);
            const simd_type neg_zeros = fyx::simd::load_brocast<simd_type>(-0.);

            simd_type t = fyx::simd::abs(v);
            simd_type sign_mask = fyx::simd::bitwise_AND(neg_zeros, v);

            t = fyx::simd::divide(ones, fyx::simd::fma(coef0, t, ones));
            simd_type r = fyx::simd::fma(coef1, t, coef2);
            r = fyx::simd::fma(r, t, coef3);
            r = fyx::simd::fma(r, t, coef4);
            r = fyx::simd::fma(r, t, coef5);

            simd_type v2 = fyx::simd::multiplies(v, v);
            simd_type mv2 = fyx::simd::bitwise_XOR(neg_zeros, v2);

            simd_type expres = fyx::simd::exp(mv2);
            simd_type neg_exp = fyx::simd::bitwise_XOR(neg_zeros, expres);
            simd_type res = fyx::simd::multiplies(t, neg_exp);
            res = fyx::simd::fma(r, res, ones);
            return fyx::simd::bitwise_XOR(sign_mask, res);
        }
    }

    float32x8 erf(float32x8 input) { return fyx::simd::detail::erf_soft_simulation<float32x8>(input); }
    float64x4 erf(float64x4 input) { return fyx::simd::detail::erf_soft_simulation<float64x4>(input); }
    float32x4 erf(float32x4 input) { return fyx::simd::detail::erf_soft_simulation<float32x4>(input); }
    float64x2 erf(float64x2 input) { return fyx::simd::detail::erf_soft_simulation<float64x2>(input); }
#endif
#if defined(_FOYE_SIMD_HAS_FP16_)
    float16x8 erf(float16x8 input)
    {
        __m256 vsrc = cvt8lane_fp16_to_fp32(input.data);
        __m256 vres32 = fyx::simd::erf(float32x8{ vsrc }).data;
        return float16x8{ cvt8lane_fp32_to_fp16(vres32) };
    }
    float16x16 erf(float16x16 input)
    {
        __m256 vsrc_low = cvt8lane_fp16_to_fp32(detail::split_low(input.data));
        __m256 vsrc_high = cvt8lane_fp16_to_fp32(detail::split_high(input.data));
        __m128i v_low = cvt8lane_fp32_to_fp16(fyx::simd::erf(float32x8{ vsrc_low }).data);
        __m128i v_high = cvt8lane_fp32_to_fp16(fyx::simd::erf(float32x8{ vsrc_high }).data);
        return float16x16{ detail::merge(v_low, v_high) };
    }
#endif
#if defined(_FOYE_SIMD_HAS_BF16_)
    bfloat16x8 erf(bfloat16x8 input)
    {
        __m256 vsrc = cvt8lane_bf16_to_fp32(input.data);
        __m256 vres32 = fyx::simd::erf(float32x8{ vsrc }).data;
        return bfloat16x8{ cvt8lane_fp32_to_bf16(vres32) };
    }
    bfloat16x16 erf(bfloat16x16 input)
    {
        __m256 vsrc_low = cvt8lane_bf16_to_fp32(detail::split_low(input.data));
        __m256 vsrc_high = cvt8lane_bf16_to_fp32(detail::split_high(input.data));
        __m128i v_low = cvt8lane_fp32_to_bf16(fyx::simd::erf(float32x8{ vsrc_low }).data);
        __m128i v_high = cvt8lane_fp32_to_bf16(fyx::simd::erf(float32x8{ vsrc_high }).data);
        return bfloat16x16{ detail::merge(v_low, v_high) };
    }
#endif

#if defined(FOYE_SIMD_ENABLE_SVML)
    float32x8 cdfnorm(float32x8 input) { return float32x8{ _mm256_cdfnorm_ps(input.data) }; }
    float64x4 cdfnorm(float64x4 input) { return float64x4{ _mm256_cdfnorm_pd(input.data) }; }
    float32x4 cdfnorm(float32x4 input) { return float32x4{ _mm_cdfnorm_ps(input.data) }; }
    float64x2 cdfnorm(float64x2 input) { return float64x2{ _mm_cdfnorm_pd(input.data) }; }
#else
    namespace detail
    {
        template<typename simd_type>
        simd_type cdfnorm_soft_simulation(simd_type v)
        {
            simd_type vtemp = fyx::simd::erf(fyx::simd::multiplies(v, 
                load_brocast<simd_type>(static_cast<typename simd_type::scalar_t>(0.7071067811865475))));
            vtemp = fyx::simd::plus(load_brocast<simd_type>(1.), vtemp);
            return fyx::simd::multiplies(load_brocast<simd_type>(0.5), vtemp);
        }
    }

    float32x8 cdfnorm(float32x8 input) { return fyx::simd::detail::cdfnorm_soft_simulation<float32x8>(input); }
    float64x4 cdfnorm(float64x4 input) { return fyx::simd::detail::cdfnorm_soft_simulation<float64x4>(input); }
    float32x4 cdfnorm(float32x4 input) { return fyx::simd::detail::cdfnorm_soft_simulation<float32x4>(input); }
    float64x2 cdfnorm(float64x2 input) { return fyx::simd::detail::cdfnorm_soft_simulation<float64x2>(input); }
#endif
#if defined(_FOYE_SIMD_HAS_FP16_)
    float16x8 cdfnorm(float16x8 input)
    {
        __m256 vsrc = cvt8lane_fp16_to_fp32(input.data);
        __m256 vres32 = fyx::simd::cdfnorm(float32x8{ vsrc }).data;
        return float16x8{ cvt8lane_fp32_to_fp16(vres32) };
    }
    float16x16 cdfnorm(float16x16 input)
    {
        __m256 vsrc_low = cvt8lane_fp16_to_fp32(detail::split_low(input.data));
        __m256 vsrc_high = cvt8lane_fp16_to_fp32(detail::split_high(input.data));
        __m128i v_low = cvt8lane_fp32_to_fp16(fyx::simd::cdfnorm(float32x8{ vsrc_low }).data);
        __m128i v_high = cvt8lane_fp32_to_fp16(fyx::simd::cdfnorm(float32x8{ vsrc_high }).data);
        return float16x16{ detail::merge(v_low, v_high) };
    }
#endif
#if defined(_FOYE_SIMD_HAS_BF16_)
    bfloat16x8 cdfnorm(bfloat16x8 input)
    {
        __m256 vsrc = cvt8lane_bf16_to_fp32(input.data);
        __m256 vres32 = fyx::simd::cdfnorm(float32x8{ vsrc }).data;
        return bfloat16x8{ cvt8lane_fp32_to_bf16(vres32) };
    }
    bfloat16x16 cdfnorm(bfloat16x16 input)
    {
        __m256 vsrc_low = cvt8lane_bf16_to_fp32(detail::split_low(input.data));
        __m256 vsrc_high = cvt8lane_bf16_to_fp32(detail::split_high(input.data));
        __m128i v_low = cvt8lane_fp32_to_bf16(fyx::simd::cdfnorm(float32x8{ vsrc_low }).data);
        __m128i v_high = cvt8lane_fp32_to_bf16(fyx::simd::cdfnorm(float32x8{ vsrc_high }).data);
        return bfloat16x16{ detail::merge(v_low, v_high) };
    }
#endif

#if defined(FOYE_SIMD_ENABLE_SVML)
    float32x8 log(float32x8 input) { return float32x8{ _mm256_log_ps(input.data) }; }
    float64x4 log(float64x4 input) { return float64x4{ _mm256_log_pd(input.data) }; }
    float32x4 log(float32x4 input) { return float32x4{ _mm_log_ps(input.data) }; }
    float64x2 log(float64x2 input) { return float64x2{ _mm_log_pd(input.data) }; }
#else
    namespace detail
    {
        template<typename simd_type>
        simd_type logFP32_soft_simulation(const simd_type& x)
        {
            using int_vector_t = fyx::simd::basic_simd<fyx::simd::detail::integral_t<
                simd_type::scalar_bit_width, true>, simd_type::bit_width>;

            const simd_type one = fyx::simd::load_brocast<simd_type>(1.0f);
            const simd_type sqrt_half = fyx::simd::load_brocast<simd_type>(0.707106781186547524f);
            const simd_type log_q1 = fyx::simd::load_brocast<simd_type>(-2.12194440E-4f);
            const simd_type log_q2 = fyx::simd::load_brocast<simd_type>(0.693359375f);

            const int_vector_t inv_mantissa_mask = load_brocast<int_vector_t>(
                typename int_vector_t::scalar_t{ ~0x7f800000 });

            int_vector_t x_bits = fyx::simd::reinterpret<int_vector_t>(x);
            int_vector_t exponent_bits = fyx::simd::shift_right<23>(x_bits);

            x_bits = fyx::simd::bitwise_AND(x_bits, inv_mantissa_mask);
            x_bits = fyx::simd::bitwise_OR(x_bits, fyx::simd::reinterpret<int_vector_t>(load_brocast<simd_type>(0.5f)));
            simd_type mantissa = fyx::simd::reinterpret<simd_type>(x_bits);

            exponent_bits = fyx::simd::minus(exponent_bits, fyx::simd::load_brocast<int_vector_t>(
                typename int_vector_t::scalar_t{ 0x7f }));
            simd_type exponent = fyx::simd::floating<simd_type>(exponent_bits);

            exponent = fyx::simd::plus(exponent, one);

            simd_type needs_adjustment = fyx::simd::less(mantissa, sqrt_half).as_basic_simd<simd_type>();

            simd_type adjustment = fyx::simd::bitwise_AND(mantissa, needs_adjustment);
            mantissa = fyx::simd::minus(mantissa, one);
            exponent = fyx::simd::minus(exponent, fyx::simd::bitwise_AND(one, needs_adjustment));
            mantissa = fyx::simd::plus(mantissa, adjustment);

            simd_type mantissa_squared = multiplies(mantissa, mantissa);

            const simd_type poly_coeff0 = fyx::simd::load_brocast<simd_type>(7.0376836292E-2f);
            const simd_type poly_coeff1 = fyx::simd::load_brocast<simd_type>(-1.1514610310E-1f);
            const simd_type poly_coeff2 = fyx::simd::load_brocast<simd_type>(1.1676998740E-1f);
            const simd_type poly_coeff3 = fyx::simd::load_brocast<simd_type>(-1.2420140846E-1f);
            const simd_type poly_coeff4 = fyx::simd::load_brocast<simd_type>(1.4249322787E-1f);
            const simd_type poly_coeff5 = fyx::simd::load_brocast<simd_type>(-1.6668057665E-1f);
            const simd_type poly_coeff6 = fyx::simd::load_brocast<simd_type>(2.0000714765E-1f);
            const simd_type poly_coeff7 = fyx::simd::load_brocast<simd_type>(-2.4999993993E-1f);
            const simd_type poly_coeff8 = fyx::simd::load_brocast<simd_type>(3.3333331174E-1f);

            simd_type polynomial = fyx::simd::fma(poly_coeff0, mantissa, poly_coeff1);
            polynomial = fyx::simd::fma(polynomial, mantissa, poly_coeff2);
            polynomial = fyx::simd::fma(polynomial, mantissa, poly_coeff3);
            polynomial = fyx::simd::fma(polynomial, mantissa, poly_coeff4);
            polynomial = fyx::simd::fma(polynomial, mantissa, poly_coeff5);
            polynomial = fyx::simd::fma(polynomial, mantissa, poly_coeff6);
            polynomial = fyx::simd::fma(polynomial, mantissa, poly_coeff7);
            polynomial = fyx::simd::fma(polynomial, mantissa, poly_coeff8);

            polynomial = fyx::simd::multiplies(polynomial, mantissa);
            polynomial = fyx::simd::multiplies(polynomial, mantissa_squared);

            polynomial = fyx::simd::fma(exponent, log_q1, polynomial);

            polynomial = fyx::simd::minus(polynomial, fyx::simd::multiplies(mantissa_squared, fyx::simd::load_brocast<simd_type>(0.5f)));

            simd_type result = fyx::simd::plus(mantissa, polynomial);
            result = fyx::simd::fma(exponent, log_q2, result);

            const simd_type zero_bits = fyx::simd::allzero_bits_as<simd_type>();
            result = fyx::simd::where_assign(
                result,
                fyx::simd::reinterpret<simd_type>(fyx::simd::load_brocast<int_vector_t>(
                    static_cast<typename int_vector_t::scalar_t>(0xff800000))),
                fyx::simd::equal(x, zero_bits));

            result = fyx::simd::where_assign(
                result,
                fyx::simd::reinterpret<simd_type>(fyx::simd::load_brocast<int_vector_t>(
                    static_cast<typename int_vector_t::scalar_t>(0x7fc00000))),
                less(x, zero_bits));

            result = fyx::simd::where_assign(
                result,
                x,
                fyx::simd::equal(x, fyx::simd::reinterpret<simd_type>(fyx::simd::load_brocast<int_vector_t>(
                    static_cast<typename int_vector_t::scalar_t>(0x7f800000)))));

            return result;
        }

        template<typename simd_type>
        simd_type logFP64_soft_simulation(const simd_type& x)
        {
            using int_vector_t = fyx::simd::basic_simd<fyx::simd::detail::integral_t<
                simd_type::scalar_bit_width, true>, simd_type::bit_width>;

            const simd_type ONE = load_brocast<simd_type>(1.0);
            const simd_type SQRT_HALF = load_brocast<simd_type>(0.7071067811865475244);

            const simd_type LOG_C0 = load_brocast<simd_type>(2.121944400546905827679e-4);
            const simd_type LOG_C1 = load_brocast<simd_type>(0.693359375);

            simd_type adjusted_x, exponent_part, numerator, denominator, result, temp, x_squared;
            int_vector_t x_bits, exponent_bits;

            const int_vector_t MANTISSA_MASK = load_brocast<int_vector_t>(
                static_cast<typename int_vector_t::scalar_t>(~0x7ff0000000000000));

            x_bits = reinterpret<int_vector_t>(x);

            exponent_bits = shift_right<52>(x_bits);

            x_bits = bitwise_AND(x_bits, MANTISSA_MASK);
            x_bits = bitwise_OR(x_bits, reinterpret<int_vector_t>(load_brocast<simd_type>(0.5)));
            adjusted_x = reinterpret<simd_type>(x_bits);

            exponent_bits = minus(exponent_bits, load_brocast<int_vector_t>(
                static_cast<typename int_vector_t::scalar_t>(0x3ff)));
            exponent_part = floating<simd_type>(exponent_bits);

            exponent_part = plus(exponent_part, ONE);

            simd_type scale_mask = less(adjusted_x, SQRT_HALF).as_basic_simd<simd_type>();
            temp = bitwise_AND(adjusted_x, scale_mask);
            adjusted_x = minus(adjusted_x, ONE);
            exponent_part = minus(exponent_part, bitwise_AND(ONE, scale_mask));
            adjusted_x = plus(adjusted_x, temp);

            x_squared = multiplies(adjusted_x, adjusted_x);

            const simd_type NUM_COEF0 = load_brocast<simd_type>(1.01875663804580931796E-4);
            const simd_type NUM_COEF1 = load_brocast<simd_type>(4.97494994976747001425E-1);
            const simd_type NUM_COEF2 = load_brocast<simd_type>(4.70579119878881725854);
            const simd_type NUM_COEF3 = load_brocast<simd_type>(1.44989225341610930846E1);
            const simd_type NUM_COEF4 = load_brocast<simd_type>(1.79368678507819816313E1);
            const simd_type NUM_COEF5 = load_brocast<simd_type>(7.70838733755885391666);

            numerator = fma(NUM_COEF0, adjusted_x, NUM_COEF1);
            numerator = fma(numerator, adjusted_x, NUM_COEF2);
            numerator = fma(numerator, adjusted_x, NUM_COEF3);
            numerator = fma(numerator, adjusted_x, NUM_COEF4);
            numerator = fma(numerator, adjusted_x, NUM_COEF5);
            numerator = multiplies(numerator, adjusted_x);
            numerator = multiplies(numerator, x_squared);

            const simd_type DEN_COEF0 = load_brocast<simd_type>(1.12873587189167450590E1);
            const simd_type DEN_COEF1 = load_brocast<simd_type>(4.52279145837532221105E1);
            const simd_type DEN_COEF2 = load_brocast<simd_type>(8.29875266912776603211E1);
            const simd_type DEN_COEF3 = load_brocast<simd_type>(7.11544750618563894466E1);
            const simd_type DEN_COEF4 = load_brocast<simd_type>(2.31251620126765340583E1);

            denominator = plus(adjusted_x, DEN_COEF0);
            denominator = fma(denominator, adjusted_x, DEN_COEF1);
            denominator = fma(denominator, adjusted_x, DEN_COEF2);
            denominator = fma(denominator, adjusted_x, DEN_COEF3);
            denominator = fma(denominator, adjusted_x, DEN_COEF4);

            result = divide(numerator, denominator);
            result = minus(result, multiplies(exponent_part, LOG_C0));
            result = minus(result, multiplies(x_squared, load_brocast<simd_type>(0.5)));

            result = plus(result, adjusted_x);
            result = fma(exponent_part, LOG_C1, result);

            result = where_assign(
                result,
                reinterpret<simd_type>(load_brocast<int_vector_t>(
                    static_cast<typename int_vector_t::scalar_t>(0xfff0000000000000))),
                equal(x, allzero_bits_as<simd_type>()));

            result = where_assign(
                result,
                reinterpret<simd_type>(load_brocast<int_vector_t>(
                    static_cast<typename int_vector_t::scalar_t>(0x7ff8000000000000))),
                less(x, allzero_bits_as<simd_type>()));

            result = where_assign(
                result,
                x,
                equal(x, reinterpret<simd_type>(load_brocast<int_vector_t>(
                    static_cast<typename int_vector_t::scalar_t>(0x7ff0000000000000)))));

            return result;
        }
    }

    float32x8 log(float32x8 input) { return fyx::simd::detail::logFP32_soft_simulation<float32x8>(input); }
    float64x4 log(float64x4 input) { return fyx::simd::detail::logFP64_soft_simulation<float64x4>(input); }
    float32x4 log(float32x4 input) { return fyx::simd::detail::logFP32_soft_simulation<float32x4>(input); }
    float64x2 log(float64x2 input) { return fyx::simd::detail::logFP64_soft_simulation<float64x2>(input); }
#endif
#if defined(_FOYE_SIMD_HAS_FP16_)
    float16x8 log(float16x8 input)
    {
        __m256 vsrc = cvt8lane_fp16_to_fp32(input.data);
        __m256 vres32 = fyx::simd::log(float32x8{ vsrc }).data;
        return float16x8{ cvt8lane_fp32_to_fp16(vres32) };
    }
    float16x16 log(float16x16 input)
    {
        __m256 vsrc_low = cvt8lane_fp16_to_fp32(detail::split_low(input.data));
        __m256 vsrc_high = cvt8lane_fp16_to_fp32(detail::split_high(input.data));
        __m128i v_low = cvt8lane_fp32_to_fp16(fyx::simd::log(float32x8{ vsrc_low }).data);
        __m128i v_high = cvt8lane_fp32_to_fp16(fyx::simd::log(float32x8{ vsrc_high }).data);
        return float16x16{ detail::merge(v_low, v_high) };
    }
#endif
#if defined(_FOYE_SIMD_HAS_BF16_)
    bfloat16x8 log(bfloat16x8 input)
    {
        __m256 vsrc = cvt8lane_bf16_to_fp32(input.data);
        __m256 vres32 = fyx::simd::log(float32x8{ vsrc }).data;
        return bfloat16x8{ cvt8lane_fp32_to_bf16(vres32) };
    }
    bfloat16x16 log(bfloat16x16 input)
    {
        __m256 vsrc_low = cvt8lane_bf16_to_fp32(detail::split_low(input.data));
        __m256 vsrc_high = cvt8lane_bf16_to_fp32(detail::split_high(input.data));
        __m128i v_low = cvt8lane_fp32_to_bf16(fyx::simd::log(float32x8{ vsrc_low }).data);
        __m128i v_high = cvt8lane_fp32_to_bf16(fyx::simd::log(float32x8{ vsrc_high }).data);
        return bfloat16x16{ detail::merge(v_low, v_high) };
    }
#endif

#if defined(FOYE_SIMD_ENABLE_SVML)
    float32x8 cbrt(float32x8 input) { return float32x8{ _mm256_cbrt_ps(input.data) }; }
    float64x4 cbrt(float64x4 input) { return float64x4{ _mm256_cbrt_pd(input.data) }; }
    float32x4 cbrt(float32x4 input) { return float32x4{ _mm_cbrt_ps(input.data) }; }
    float64x2 cbrt(float64x2 input) { return float64x2{ _mm_cbrt_pd(input.data) }; }
#else
    namespace detail
    {
        template<typename simd_type>
        simd_type cuberootF32_soft_simulation(simd_type x)
        {
            constexpr float epsilon = std::numeric_limits<float>::epsilon();

            const simd_type zero = load_brocast<simd_type>(0.0f);
            const simd_type two = load_brocast<simd_type>(2.0f);
            const simd_type three = load_brocast<simd_type>(3.0f);
            const simd_type eps = load_brocast<simd_type>(epsilon);
            const simd_type nan = load_brocast<simd_type>(std::numeric_limits<float>::quiet_NaN());
            const simd_type nzero = load_brocast<simd_type>(-0.0f);

            mask_from_simd_t<simd_type> is_zero = equal(bitwise_ANDNOT(nzero, x), zero);
            mask_from_simd_t<simd_type> is_nan = not_equal(x, x);
            mask_from_simd_t<simd_type> is_inf = equal(bitwise_ANDNOT(nzero, x), 
                load_brocast<simd_type>(std::numeric_limits<float>::infinity()));

            simd_type sign_mask = bitwise_AND(x, nzero);
            simd_type abs_x = bitwise_ANDNOT(sign_mask, x);
            const simd_type neone = load_brocast<simd_type>(-1.f);

            simd_type guess = abs_x;

            for (int iter = 0; iter < 50; ++iter)
            {
                simd_type guess_squared = multiplies(guess, guess);

                mask_from_simd_t<simd_type> guess_sq_zero = equal(guess_squared, zero);
                if (bitwise_test_check(guess_sq_zero.as_basic_simd<simd_type>(), neone))
                {
                    break;
                }

                simd_type x_div_guess_sq = divide(abs_x, guess_squared);
                simd_type two_guess = multiplies(two, guess);
                simd_type numerator = plus(two_guess, x_div_guess_sq);
                simd_type next_guess = divide(numerator, three);

                simd_type diff = minus(next_guess, guess);
                simd_type abs_diff = bitwise_ANDNOT(nzero, diff);
                simd_type abs_next_guess = bitwise_ANDNOT(nzero, next_guess);
                simd_type threshold = multiplies(eps, abs_next_guess);

                mask_from_simd_t<simd_type> converged = less(abs_diff, threshold);
                if (bitwise_test_check(converged.as_basic_simd<simd_type>(), neone))
                {
                    break;
                }

                guess = next_guess;
            }

            simd_type result = bitwise_OR(guess, sign_mask);

            result = where_assign(result, zero, is_zero);
            result = where_assign(result, nan, is_nan);
            result = where_assign(result, x, is_inf);

            return result;
        }

        template<typename simd_type>
        simd_type cuberootF64_soft_simulation(simd_type x)
        {
            constexpr double epsilon = std::numeric_limits<double>::epsilon();

            const simd_type zero = load_brocast<simd_type>(0.0);
            const simd_type two = load_brocast<simd_type>(2.0);
            const simd_type three = load_brocast<simd_type>(3.0);
            const simd_type eps = load_brocast<simd_type>(epsilon);
            const simd_type nan = load_brocast<simd_type>(std::numeric_limits<double>::quiet_NaN());

            mask_from_simd_t<simd_type> is_zero = equal(bitwise_ANDNOT(load_brocast<simd_type>(-0.0), x), zero);
            mask_from_simd_t<simd_type> is_nan = not_equal(x, x);
            mask_from_simd_t<simd_type> is_inf = equal(bitwise_ANDNOT(load_brocast<simd_type>(-0.0), x),
                load_brocast<simd_type>(std::numeric_limits<double>::infinity()));

            simd_type sign_mask = bitwise_AND(x, load_brocast<simd_type>(-0.0));
            simd_type abs_x = bitwise_ANDNOT(sign_mask, x);

            simd_type guess = abs_x;

            for (int iter = 0; iter < 50; ++iter)
            {
                simd_type guess_squared = multiplies(guess, guess);

                mask_from_simd_t<simd_type> guess_sq_zero = equal(guess_squared, zero);

                bool allzero{};
                if constexpr (fyx::simd::is_256bits_simd_v<simd_type>)
                {
                    allzero = _mm256_movemask_pd(detail::basic_reinterpret<__m256d>(
                        guess_sq_zero.data)) == 0xF;
                }
                else
                {
                    allzero = _mm_movemask_pd(detail::basic_reinterpret<__m128d>(
                        guess_sq_zero.data)) == 0xF;
                }

                if (allzero)
                {
                    break;
                }

                simd_type x_div_guess_sq = divide(abs_x, guess_squared);
                simd_type two_guess = multiplies(two, guess);
                simd_type numerator = plus(two_guess, x_div_guess_sq);
                simd_type next_guess = divide(numerator, three);

                simd_type diff = minus(next_guess, guess);
                simd_type abs_diff = bitwise_ANDNOT(load_brocast<simd_type>(-0.0), diff);
                simd_type abs_next_guess = bitwise_ANDNOT(load_brocast<simd_type>(-0.0), next_guess);
                simd_type threshold = multiplies(eps, abs_next_guess);

                mask_from_simd_t<simd_type> converged = less(abs_diff, threshold);
                int all_convergence{};
                if constexpr (fyx::simd::is_256bits_simd_v<simd_type>)
                {
                    all_convergence = (_mm256_movemask_pd(detail::basic_reinterpret<__m256d>(
                        converged.data)) == 0xF);
                }
                else
                {
                    all_convergence = (_mm_movemask_pd(detail::basic_reinterpret<__m128d>(
                        converged.data)) == 0xF);
                }

                if (all_convergence)
                {
                    guess = next_guess;
                    break;
                }

                guess = next_guess;
            }

            simd_type result = bitwise_OR(guess, sign_mask);

            result = where_assign(result, zero, is_zero);
            result = where_assign(result, nan, is_nan);
            result = where_assign(result, x, is_inf);

            return result;
        }
    }

    float32x8 cbrt(float32x8 input) { return fyx::simd::detail::cuberootF32_soft_simulation(input); }
    float64x4 cbrt(float64x4 input) { return fyx::simd::detail::cuberootF64_soft_simulation(input); }
    float32x4 cbrt(float32x4 input) { return fyx::simd::detail::cuberootF32_soft_simulation(input); }
    float64x2 cbrt(float64x2 input) { return fyx::simd::detail::cuberootF64_soft_simulation(input); }
#endif
#if defined(_FOYE_SIMD_HAS_FP16_)
    float16x8 cbrt(float16x8 input)
    {
        __m256 vsrc = cvt8lane_fp16_to_fp32(input.data);
        __m256 vres32 = fyx::simd::cbrt(float32x8{ vsrc }).data;
        return float16x8{ cvt8lane_fp32_to_fp16(vres32) };
    }
    float16x16 cbrt(float16x16 input)
    {
        __m256 vsrc_low = cvt8lane_fp16_to_fp32(detail::split_low(input.data));
        __m256 vsrc_high = cvt8lane_fp16_to_fp32(detail::split_high(input.data));
        __m128i v_low = cvt8lane_fp32_to_fp16(fyx::simd::cbrt(float32x8{ vsrc_low }).data);
        __m128i v_high = cvt8lane_fp32_to_fp16(fyx::simd::cbrt(float32x8{ vsrc_high }).data);
        return float16x16{ detail::merge(v_low, v_high) };
    }
#endif
#if defined(_FOYE_SIMD_HAS_BF16_)
    bfloat16x8 cbrt(bfloat16x8 input)
    {
        __m256 vsrc = cvt8lane_bf16_to_fp32(input.data);
        __m256 vres32 = fyx::simd::cbrt(float32x8{ vsrc }).data;
        return bfloat16x8{ cvt8lane_fp32_to_bf16(vres32) };
    }
    bfloat16x16 cbrt(bfloat16x16 input)
    {
        __m256 vsrc_low = cvt8lane_bf16_to_fp32(detail::split_low(input.data));
        __m256 vsrc_high = cvt8lane_bf16_to_fp32(detail::split_high(input.data));
        __m128i v_low = cvt8lane_fp32_to_bf16(fyx::simd::cbrt(float32x8{ vsrc_low }).data);
        __m128i v_high = cvt8lane_fp32_to_bf16(fyx::simd::cbrt(float32x8{ vsrc_high }).data);
        return bfloat16x16{ detail::merge(v_low, v_high) };
    }
#endif

#if defined(FOYE_SIMD_ENABLE_SVML)
    float32x8 pow(float32x8 arg0, float32x8 arg1) { return float32x8{ _mm256_pow_ps(arg0.data, arg1.data) }; }
    float64x4 pow(float64x4 arg0, float64x4 arg1) { return float64x4{ _mm256_pow_pd(arg0.data, arg1.data) }; }
    float32x4 pow(float32x4 arg0, float32x4 arg1) { return float32x4{ _mm_pow_ps(arg0.data, arg1.data) }; }
    float64x2 pow(float64x2 arg0, float64x2 arg1) { return float64x2{ _mm_pow_pd(arg0.data, arg1.data) }; }

    float32x8 pow(float32x8 arg0, sint32x8 arg1) { return fyx::simd::pow(arg0, fyx::simd::floating<float32x8>(arg1)); }
    float64x4 pow(float64x4 arg0, sint64x4 arg1) { return fyx::simd::pow(arg0, fyx::simd::floating<float64x4>(arg1)); }
    float32x4 pow(float32x4 arg0, sint32x4 arg1) { return fyx::simd::pow(arg0, fyx::simd::floating<float32x4>(arg1)); }
    float64x2 pow(float64x2 arg0, sint64x2 arg1) { return fyx::simd::pow(arg0, fyx::simd::floating<float64x2>(arg1)); }
#else
    namespace detail
    {
        template<typename simd_type>
        simd_type pow_iexponent_soft_simulation(simd_type base,
            basic_simd<detail::integral_t<simd_type::scalar_bit_width, true>, simd_type::bit_width> exponent)
        {
            using sint_simd_t = basic_simd<detail::integral_t<
                simd_type::scalar_bit_width, true>, simd_type::bit_width>;

            simd_type result = load_brocast<simd_type>(1.0f);
            simd_type baseraw = base;

            sint_simd_t abs_n = abs(exponent);

            sint_simd_t mask = load_brocast<sint_simd_t>(1);
            const sint_simd_t neone = load_brocast<sint_simd_t>(-1);

            for (int i = 0; i < 32; ++i) 
            {
                sint_simd_t bit = bitwise_AND(abs_n, mask);
                simd_type condition = floating<simd_type>(equal(bit, mask).as_basic_simd<sint_simd_t>());

                result = where_assign(result, multiplies(result, baseraw), condition);
                baseraw = multiplies(baseraw, baseraw);
                abs_n = reinterpret<sint_simd_t>(shift_right<1>(
                    reinterpret<as_unsigned_type<sint_simd_t>>(abs_n)));

                if (bitwise_test_zero(abs_n, neone))
                {
                    break;
                }
            }

            simd_type reciprocal = rcp(result);
            result = where_assign(result, reciprocal,
                greater(allzero_bits_as<sint_simd_t>(), exponent));

            return result;
        }

        template<typename simd_type>
        simd_type powF32_fexponent_soft_simulation(simd_type base, simd_type exponent)
        {
            const mask_from_simd_t<simd_type> zero_mask = equal(base, allzero_bits_as<simd_type>());
            const mask_from_simd_t<simd_type> one_mask = equal(base, load_brocast<simd_type>(1.0));
            const mask_from_simd_t<simd_type> neg_mask = less(base, allzero_bits_as<simd_type>());

            simd_type result_one = load_brocast<simd_type>(1.0);

            mask_from_simd_t<simd_type> pos_exp_mask = greater(exponent, allzero_bits_as<simd_type>());
            simd_type zero_result = allzero_bits_as<simd_type>();
            simd_type inf_result = load_brocast<simd_type>(std::numeric_limits<typename simd_type::scalar_t>::infinity());
            simd_type result_zero = where_assign(inf_result, zero_result, pos_exp_mask);

            using sint_simd_t = basic_simd<detail::integral_t<
                simd_type::scalar_bit_width, true>, simd_type::bit_width>;

            sint_simd_t exp_int = trunc_as_i(exponent);
            simd_type exp_float = floating<simd_type>(exp_int);
            mask_from_simd_t<simd_type> int_mask = equal(exponent, exp_float);

            simd_type nan_result = load_brocast<simd_type>(std::numeric_limits<typename simd_type::scalar_t>::quiet_NaN());
            simd_type neg_valid = bitwise_AND(neg_mask.as_basic_simd<simd_type>(), int_mask.as_basic_simd<simd_type>());

            simd_type abs_base = bitwise_AND(base, reinterpret<simd_type>(load_brocast<sint_simd_t>(0x7FFFFFFF)));
            simd_type log_abs = log(abs_base);
            simd_type exp_result = exp(multiplies(exponent, log_abs));

            sint_simd_t exp_mod2 = bitwise_AND(exp_int, load_brocast<sint_simd_t>(1));
            simd_type sign = reinterpret<simd_type>(shift_left<31>(reinterpret<as_unsigned_type<sint_simd_t>>(exp_mod2)));
            simd_type signed_result = bitwise_OR(exp_result, sign);
            
            simd_type result = where_assign(exp_result, signed_result, mask_from_simd_t<simd_type>{ neg_valid });
            result = where_assign(result, result_zero, zero_mask);
            result = where_assign(result, result_one, one_mask);
            result = where_assign(result, nan_result,
                bitwise_ANDNOT(
                    int_mask.as_basic_simd<simd_type>(), 
                    neg_mask.as_basic_simd<simd_type>()).as_basic_mask()
            );

            return result;
        }

        template<typename simd_type>
        simd_type powF64_fexponent_soft_simulation(simd_type base, simd_type exponent)
        {
            const simd_type zero = load_brocast<simd_type>(0.0);
            const simd_type one = load_brocast<simd_type>(1.0);
            const simd_type nan = load_brocast<simd_type>(std::numeric_limits<typename simd_type::scalar_t>::quiet_NaN());
            const simd_type inf = load_brocast<simd_type>(std::numeric_limits<typename simd_type::scalar_t>::infinity());

            simd_type result = nan;

            mask_from_simd_t<simd_type> mask_base_zero = equal(base, zero);
            mask_from_simd_t<simd_type> mask_exp_zero = equal(exponent, zero);
            mask_from_simd_t<simd_type> mask_base_one = equal(base, one);
            mask_from_simd_t<simd_type> mask_base_neg = less(base, zero);

            result = where_assign(result, one, mask_base_one);

            simd_type mask_zero_zero = bitwise_AND(
                mask_base_zero.as_basic_simd<simd_type>(), 
                mask_exp_zero.as_basic_simd<simd_type>());

            result = where_assign(result, one, mask_from_simd_t<simd_type>{ mask_zero_zero });

            mask_from_simd_t<simd_type> mask_pos_exp = greater(exponent, zero);
            simd_type zero_to_pos = bitwise_AND(
                mask_base_zero.as_basic_simd<simd_type>(), 
                mask_pos_exp.as_basic_simd<simd_type>());

            result = where_assign(result, zero, mask_from_simd_t<simd_type>{ zero_to_pos });

            mask_from_simd_t<simd_type> mask_neg_exp = less(exponent, zero);
            simd_type zero_to_neg = bitwise_AND(
                mask_base_zero.as_basic_simd<simd_type>(), 
                mask_neg_exp.as_basic_simd<simd_type>());

            result = where_assign(result, inf, mask_from_simd_t<simd_type>{ zero_to_neg });

            simd_type exp_int = round(exponent);
            mask_from_simd_t<simd_type> mask_exp_not_int = not_equal(exponent, exp_int);
            simd_type neg_base_non_int_exp = bitwise_AND(
                mask_base_neg.as_basic_simd<simd_type>(), 
                mask_exp_not_int.as_basic_simd<simd_type>());

            result = where_assign(result, nan, mask_from_simd_t<simd_type>{ neg_base_non_int_exp });

            mask_from_simd_t<simd_type> normal_mask = not_nan(result);
            normal_mask = mask_from_simd_t<simd_type>{ bitwise_NOT(normal_mask.as_basic_simd<simd_type>()) };

            int movedmask{};
            if constexpr (fyx::simd::is_256bits_simd_v<simd_type>)
            {
                movedmask = _mm256_movemask_pd(
                    detail::basic_reinterpret<__m256d>(normal_mask.data));
            }
            else
            {
                movedmask = _mm_movemask_pd(
                    detail::basic_reinterpret<__m128d>(normal_mask.data));
            }

            if (movedmask != 0)
            {
                simd_type safe_base = where_assign(base, one, mask_base_zero);
                safe_base = where_assign(safe_base, one, mask_base_neg);

                simd_type log_base = log(safe_base);
                simd_type product = multiplies(exponent, log_base);
                simd_type normal_result = exp(product);

                result = where_assign(result, normal_result, normal_mask);
            }

            return result;
        }
    }

    float32x8 pow(float32x8 base, sint32x8 exponent) { return detail::pow_iexponent_soft_simulation(base, exponent); }
    float64x4 pow(float64x4 base, sint64x4 exponent) { return detail::pow_iexponent_soft_simulation(base, exponent); }
    float32x4 pow(float32x4 base, sint32x4 exponent) { return detail::pow_iexponent_soft_simulation(base, exponent); }
    float64x2 pow(float64x2 base, sint64x2 exponent) { return detail::pow_iexponent_soft_simulation(base, exponent); }

    float32x8 pow(float32x8 base, float32x8 exponent) { return detail::powF32_fexponent_soft_simulation(base, exponent); }
    float64x4 pow(float64x4 base, float64x4 exponent) { return detail::powF64_fexponent_soft_simulation(base, exponent); }
    float32x4 pow(float32x4 base, float32x4 exponent) { return detail::powF32_fexponent_soft_simulation(base, exponent); }
    float64x2 pow(float64x2 base, float64x2 exponent) { return detail::powF64_fexponent_soft_simulation(base, exponent); }
#endif
#if defined(_FOYE_SIMD_HAS_FP16_)
    float16x8 pow(float16x8 base, sint16x8 exponent)
    {
        float32x8 base32 = fyx::simd::expand<float32x8>(base);
        sint32x8 exponent32 = fyx::simd::expand<sint32x8>(exponent);
        float32x8 result32 = fyx::simd::pow(base32, exponent32);
        return fyx::simd::narrowing<float16x8>(result32);
    }

    float16x16 pow(float16x16 base, sint16x16 exponent)
    {
        float32x8 result_low = fyx::simd::pow(
            fyx::simd::expand_low<float32x8>(base),
            fyx::simd::expand_low<sint32x8>(exponent));

        float32x8 result_high = fyx::simd::pow(
            fyx::simd::expand_high<float32x8>(base),
            fyx::simd::expand_high<sint32x8>(exponent));

        return fyx::simd::merge(
            fyx::simd::narrowing<float16x8>(result_low),
            fyx::simd::narrowing<float16x8>(result_high)
        );
    }

    float16x8 pow(float16x8 base, float16x8 exponent)
    {
        float32x8 base32 = fyx::simd::expand<float32x8>(base);
        float32x8 exponent32 = fyx::simd::expand<float32x8>(exponent);
        float32x8 result32 = fyx::simd::pow(base32, exponent32);
        return fyx::simd::narrowing<float16x8>(result32);
    }

    float16x16 pow(float16x16 base, float16x16 exponent)
    {
        float32x8 result_low = fyx::simd::pow(
            fyx::simd::expand_low<float32x8>(base),
            fyx::simd::expand_low<float32x8>(exponent));

        float32x8 result_high = fyx::simd::pow(
            fyx::simd::expand_high<float32x8>(base),
            fyx::simd::expand_high<float32x8>(exponent));

        return fyx::simd::merge(
            fyx::simd::narrowing<float16x8>(result_low),
            fyx::simd::narrowing<float16x8>(result_high)
        );
    }
#endif
#if defined(_FOYE_SIMD_HAS_BF16_)
    bfloat16x8 pow(bfloat16x8 base, sint16x8 exponent)
    {
        float32x8 base32 = fyx::simd::expand<float32x8>(base);
        sint32x8 exponent32 = fyx::simd::expand<sint32x8>(exponent);
        float32x8 result32 = fyx::simd::pow(base32, exponent32);
        return fyx::simd::narrowing<bfloat16x8>(result32);
    }

    bfloat16x16 pow(bfloat16x16 base, sint16x16 exponent)
    {
        float32x8 result_low = fyx::simd::pow(
            fyx::simd::expand_low<float32x8>(base),
            fyx::simd::expand_low<sint32x8>(exponent));

        float32x8 result_high = fyx::simd::pow(
            fyx::simd::expand_high<float32x8>(base),
            fyx::simd::expand_high<sint32x8>(exponent));

        return fyx::simd::merge(
            fyx::simd::narrowing<bfloat16x8>(result_low),
            fyx::simd::narrowing<bfloat16x8>(result_high)
        );
    }

    bfloat16x8 pow(bfloat16x8 base, bfloat16x8 exponent)
    {
        float32x8 base32 = fyx::simd::expand<float32x8>(base);
        float32x8 exponent32 = fyx::simd::expand<float32x8>(exponent);
        float32x8 result32 = fyx::simd::pow(base32, exponent32);
        return fyx::simd::narrowing<bfloat16x8>(result32);
    }

    bfloat16x16 pow(bfloat16x16 base, bfloat16x16 exponent)
    {
        float32x8 result_low = fyx::simd::pow(
            fyx::simd::expand_low<float32x8>(base),
            fyx::simd::expand_low<float32x8>(exponent));

        float32x8 result_high = fyx::simd::pow(
            fyx::simd::expand_high<float32x8>(base),
            fyx::simd::expand_high<float32x8>(exponent));

        return fyx::simd::merge(
            fyx::simd::narrowing<bfloat16x8>(result_low),
            fyx::simd::narrowing<bfloat16x8>(result_high)
        );
    }
#endif

#if defined(FOYE_SIMD_ENABLE_SVML)
    float32x8 logb(float32x8 input) { return float32x8{ _mm256_logb_ps(input.data) }; }
    float64x4 logb(float64x4 input) { return float64x4{ _mm256_logb_pd(input.data) }; }
    float32x4 logb(float32x4 input) { return float32x4{ _mm_logb_ps(input.data) }; }
    float64x2 logb(float64x2 input) { return float64x2{ _mm_logb_pd(input.data) }; }
#else
    namespace detail
    {
        template<typename simd_type>
        simd_type logbF32_fexponent_soft_simulation(simd_type input)
        {
            using sint_simd_t = basic_simd<detail::integral_t<
                simd_type::scalar_bit_width, true>, simd_type::bit_width>;

            sint_simd_t x_int = reinterpret<sint_simd_t>(input);

            sint_simd_t exponent_bits = shift_right<23>(x_int);
            sint_simd_t exponent = bitwise_AND(exponent_bits, load_brocast<sint_simd_t>(0xFF));

            mask_from_simd_t<simd_type> is_zero = equal(input, allzero_bits_as<simd_type>());

            simd_type is_denormal = bitwise_AND(
                equal(reinterpret<simd_type>(exponent), allzero_bits_as<simd_type>()).as_basic_simd<simd_type>(),
                not_equal(input, allzero_bits_as<simd_type>()).as_basic_simd<simd_type>()
            );

            sint_simd_t mantissa = bitwise_AND(x_int, load_brocast<sint_simd_t>(0x007FFFFF));

            simd_type result_normal = floating<simd_type>(minus(exponent, load_brocast<sint_simd_t>(127)));

            const simd_type minus_inf = load_brocast<simd_type>(-(std::numeric_limits<typename simd_type::scalar_t>::infinity()));
            const simd_type plus_inf = load_brocast<simd_type>(std::numeric_limits<typename simd_type::scalar_t>::infinity());
            const simd_type nan_val = load_brocast<simd_type>(std::numeric_limits<typename simd_type::scalar_t>::quiet_NaN());

            result_normal = where_assign(result_normal, minus_inf, is_zero);

            mask_from_simd_t<simd_type> is_inf_nan = equal(reinterpret<simd_type>(exponent),
                reinterpret<simd_type>(load_brocast<sint_simd_t>(0xFF)));

            simd_type is_nan = bitwise_AND(is_inf_nan.as_basic_simd<simd_type>(),
                not_equal(reinterpret<simd_type>(mantissa),
                    allzero_bits_as<simd_type>()).as_basic_simd<simd_type>());

            simd_type is_inf = bitwise_AND(is_inf_nan.as_basic_simd<simd_type>(),
                equal(reinterpret<simd_type>(mantissa),
                    allzero_bits_as<simd_type>()).as_basic_simd<simd_type>());


            result_normal = where_assign(result_normal, plus_inf, is_inf.as_basic_mask());
            result_normal = where_assign(result_normal, nan_val, is_nan.as_basic_mask());

            simd_type denormal_scaled = multiplies(input, load_brocast<simd_type>(8388608.0f));
            sint_simd_t denormal_scaled_int = reinterpret<sint_simd_t>(denormal_scaled);
            sint_simd_t denormal_exponent_bits = shift_right<23>(denormal_scaled_int);
            sint_simd_t denormal_exponent = bitwise_AND(denormal_exponent_bits, load_brocast<sint_simd_t>(0xFF));

            simd_type result_denormal = minus(
                floating<simd_type>(minus(denormal_exponent, load_brocast<sint_simd_t>(127))),
                load_brocast<simd_type>(23.0f)
            );

            simd_type result = where_assign(result_normal, result_denormal, is_denormal.as_basic_mask());
            return result;
        }

        template<typename simd_type>
        simd_type logbF64_fexponent_soft_simulation(simd_type input)
        {
            using sint_simd_t = basic_simd<detail::integral_t<
                simd_type::scalar_bit_width, true>, simd_type::bit_width>;

            sint_simd_t x_int = reinterpret<sint_simd_t>(input);

            sint_simd_t exponent_bits = shift_right<52>(x_int);
            sint_simd_t exponent = bitwise_AND(exponent_bits, load_brocast<sint_simd_t>(0x7FF));

            mask_from_simd_t<simd_type> is_zero = equal(input, allzero_bits_as<simd_type>());

            simd_type is_denormal = bitwise_AND(
                equal(reinterpret<simd_type>(exponent), allzero_bits_as<simd_type>()).as_basic_simd<simd_type>(),
                not_equal(input, allzero_bits_as<simd_type>()).as_basic_simd<simd_type>()
            );

            simd_type result_normal = floating<simd_type>(minus(exponent, load_brocast<sint_simd_t>(1023)));

            simd_type minus_inf = load_brocast<simd_type>(-(std::numeric_limits<typename simd_type::scalar_t>::infinity()));
            simd_type plus_inf = load_brocast<simd_type>(std::numeric_limits<typename simd_type::scalar_t>::infinity());
            simd_type nan_val = load_brocast<simd_type>(std::numeric_limits<typename simd_type::scalar_t>::quiet_NaN());

            result_normal = where_assign(result_normal, minus_inf, is_zero);

            mask_from_simd_t<simd_type> is_inf_nan = equal(reinterpret<simd_type>(exponent),
                reinterpret<simd_type>(load_brocast<sint_simd_t>(0x7FF)));

            sint_simd_t mantissa = bitwise_AND(x_int, load_brocast<sint_simd_t>(0x000FFFFFFFFFFFFF));

            simd_type is_nan = bitwise_AND(is_inf_nan.as_basic_simd<simd_type>(),
                not_equal(reinterpret<simd_type>(mantissa), allzero_bits_as<simd_type>()).as_basic_simd<simd_type>());

            simd_type is_inf = bitwise_AND(is_inf_nan.as_basic_simd<simd_type>(),
                equal(reinterpret<simd_type>(mantissa), allzero_bits_as<simd_type>()).as_basic_simd<simd_type>());

            result_normal = where_assign(result_normal, plus_inf, is_inf.as_basic_mask());
            result_normal = where_assign(result_normal, nan_val, is_nan.as_basic_mask());

            simd_type denormal_scaled = multiplies(input, load_brocast<simd_type>(4503599627370496.0));
            sint_simd_t denormal_scaled_int = reinterpret<sint_simd_t>(denormal_scaled);
            sint_simd_t denormal_exponent_bits = shift_right<52>(denormal_scaled_int);
            sint_simd_t denormal_exponent = bitwise_AND(denormal_exponent_bits, load_brocast<sint_simd_t>(0x7FF));

            simd_type result_denormal = minus(
                floating<simd_type>(minus(denormal_exponent, load_brocast<sint_simd_t>(1023))),
                load_brocast<simd_type>(52.0)
            );

            simd_type result = where_assign(result_normal, result_denormal, is_denormal.as_basic_mask());
            return result;
        }
    }

    float32x8 logb(float32x8 input) { return fyx::simd::detail::logbF32_fexponent_soft_simulation(input); }
    float32x4 logb(float32x4 input) { return fyx::simd::detail::logbF32_fexponent_soft_simulation(input); }
    float64x4 logb(float64x4 input) { return fyx::simd::detail::logbF64_fexponent_soft_simulation(input); }
    float64x2 logb(float64x2 input) { return fyx::simd::detail::logbF64_fexponent_soft_simulation(input); }
#endif

#if defined(FOYE_SIMD_ENABLE_SVML)
    float32x8 sin(float32x8 input) { return float32x8{ _mm256_sin_ps(input.data) }; }
    float64x4 sin(float64x4 input) { return float64x4{ _mm256_sin_pd(input.data) }; }
    float32x4 sin(float32x4 input) { return float32x4{ _mm_sin_ps(input.data) }; }
    float64x2 sin(float64x2 input) { return float64x2{ _mm_sin_pd(input.data) }; }

    float32x8 cos(float32x8 input) { return float32x8{ _mm256_cos_ps(input.data) }; }
    float64x4 cos(float64x4 input) { return float64x4{ _mm256_cos_pd(input.data) }; }
    float32x4 cos(float32x4 input) { return float32x4{ _mm_cos_ps(input.data) }; }
    float64x2 cos(float64x2 input) { return float64x2{ _mm_cos_pd(input.data) }; }
#else
    namespace detail
    {
        constexpr std::uint32_t SIGN_BIT_MASK = 0x80000000;
        constexpr std::uint32_t INV_SIGN_BIT_MASK = ~SIGN_BIT_MASK;

        constexpr float PI_OVER_4_RECIPROCAL = 1.27323954473516f;

        constexpr float REDUCE_COEFF1 = -0.78515625f;
        constexpr float REDUCE_COEFF2 = -2.4187564849853515625e-4f;
        constexpr float REDUCE_COEFF3 = -3.77489497744594108e-8f;

        constexpr float COS_COEFF0 = 2.443315711809948E-005f;
        constexpr float COS_COEFF1 = -1.388731625493765E-003f;
        constexpr float COS_COEFF2 = 4.166664568298827E-002f;
        constexpr float HALF = 0.5f;
        constexpr float ONE = 1.0f;

        constexpr float SIN_COEFF0 = -1.9515295891E-4f;
        constexpr float SIN_COEFF1 = 8.3321608736E-3f;
        constexpr float SIN_COEFF2 = -1.6666654611E-1f;

        constexpr int INTEGER_ONE = 1;
        constexpr int INTEGER_TWO = 2;
        constexpr int INTEGER_FOUR = 4;
        constexpr int SHIFT_29 = 29;

        template<typename simd_type>
        simd_type sin_soft_simulation(simd_type input)
        {
            using sint_simd_t = basic_simd<detail::integral_t<
                simd_type::scalar_bit_width, true>, simd_type::bit_width>;

            simd_type poly_result, sign_bit, normalized_x;
            sint_simd_t quadrant_int, quadrant_parity;

            simd_type x = input;
            sign_bit = x;

            simd_type inv_sign_mask = load_brocast<simd_type>(std::bit_cast<float>(INV_SIGN_BIT_MASK));
            simd_type sign_mask = load_brocast<simd_type>(std::bit_cast<float>(SIGN_BIT_MASK));
            simd_type pi_over_4_reciprocal = load_brocast<simd_type>(PI_OVER_4_RECIPROCAL);

            x = bitwise_AND(x, inv_sign_mask);
            sign_bit = bitwise_AND(sign_bit, sign_mask);

            normalized_x = multiplies(x, pi_over_4_reciprocal);

            quadrant_int = trunc_as_i(normalized_x);

            quadrant_int = plus(quadrant_int, load_brocast<sint_simd_t>(INTEGER_ONE));
            quadrant_int = bitwise_AND(quadrant_int, load_brocast<sint_simd_t>(~INTEGER_ONE));
            normalized_x = floating<simd_type>(quadrant_int);

            quadrant_parity = bitwise_AND(quadrant_int, load_brocast<sint_simd_t>(INTEGER_FOUR));
            quadrant_parity = shift_left<SHIFT_29>(quadrant_parity);

            quadrant_int = bitwise_AND(quadrant_int, load_brocast<sint_simd_t>(INTEGER_TWO));
            quadrant_int = equal(quadrant_int, allzero_bits_as<sint_simd_t>()).as_basic_simd<sint_simd_t>();

            simd_type sign_swap_mask = reinterpret<simd_type>(quadrant_parity);
            simd_type use_cosine_mask = reinterpret<simd_type>(quadrant_int);
            sign_bit = bitwise_XOR(sign_bit, sign_swap_mask);

            simd_type reduce_term1 = load_brocast<simd_type>(REDUCE_COEFF1);
            simd_type reduce_term2 = load_brocast<simd_type>(REDUCE_COEFF2);
            simd_type reduce_term3 = load_brocast<simd_type>(REDUCE_COEFF3);

            x = fma(normalized_x, reduce_term1, x);
            x = fma(normalized_x, reduce_term2, x);
            x = fma(normalized_x, reduce_term3, x);

            simd_type cos_poly = load_brocast<simd_type>(COS_COEFF0);
            simd_type x_squared = multiplies(x, x);

            cos_poly = fma(cos_poly, x_squared, load_brocast<simd_type>(COS_COEFF1));
            cos_poly = fma(cos_poly, x_squared, load_brocast<simd_type>(COS_COEFF2));
            cos_poly = multiplies(cos_poly, x_squared);
            cos_poly = multiplies(cos_poly, x_squared);

            simd_type half_x_squared = multiplies(x_squared, load_brocast<simd_type>(HALF));
            cos_poly = minus(cos_poly, half_x_squared);
            cos_poly = plus(cos_poly, load_brocast<simd_type>(ONE));

            simd_type sin_poly = load_brocast<simd_type>(SIN_COEFF0);
            sin_poly = fma(sin_poly, x_squared, load_brocast<simd_type>(SIN_COEFF1));
            sin_poly = fma(sin_poly, x_squared, load_brocast<simd_type>(SIN_COEFF2));
            sin_poly = multiplies(sin_poly, x_squared);
            sin_poly = multiplies(sin_poly, x);
            sin_poly = plus(sin_poly, x);

            sin_poly = bitwise_AND(use_cosine_mask, sin_poly);
            cos_poly = bitwise_ANDNOT(use_cosine_mask, cos_poly);
            poly_result = plus(cos_poly, sin_poly);

            poly_result = bitwise_XOR(poly_result, sign_bit);
            return poly_result;
        }

        template<typename simd_type>
        simd_type cos_soft_simulation(simd_type input)
        {
            using sint_simd_t = basic_simd<detail::integral_t<
                simd_type::scalar_bit_width, true>, simd_type::bit_width>;

            simd_type x = input;

            simd_type inv_sign_mask = reinterpret<simd_type>(load_brocast<sint_simd_t>(INV_SIGN_BIT_MASK));
            simd_type cephes_FOPI = load_brocast<simd_type>(PI_OVER_4_RECIPROCAL);

            sint_simd_t pi32_256_1 = load_brocast<sint_simd_t>(INTEGER_ONE);
            sint_simd_t pi32_256_inv1 = load_brocast<sint_simd_t>(~INTEGER_ONE);
            sint_simd_t pi32_256_4 = load_brocast<sint_simd_t>(INTEGER_FOUR);
            sint_simd_t pi32_256_2 = load_brocast<sint_simd_t>(INTEGER_TWO);
            sint_simd_t pi32_256_0 = load_brocast<sint_simd_t>(0);

            simd_type minus_cephes_DP1 = load_brocast<simd_type>(REDUCE_COEFF1);
            simd_type minus_cephes_DP2 = load_brocast<simd_type>(REDUCE_COEFF2);
            simd_type minus_cephes_DP3 = load_brocast<simd_type>(REDUCE_COEFF3);

            simd_type sincof_p0 = load_brocast<simd_type>(SIN_COEFF0);
            simd_type sincof_p1 = load_brocast<simd_type>(SIN_COEFF1);
            simd_type sincof_p2 = load_brocast<simd_type>(SIN_COEFF2);

            simd_type coscof_p0 = load_brocast<simd_type>(COS_COEFF0);
            simd_type coscof_p1 = load_brocast<simd_type>(COS_COEFF1);
            simd_type coscof_p2 = load_brocast<simd_type>(COS_COEFF2);

            simd_type ps256_0p5 = load_brocast<simd_type>(HALF);
            simd_type ps256_1 = load_brocast<simd_type>(ONE);

            x = bitwise_AND(x, inv_sign_mask);
            simd_type y = multiplies(x, cephes_FOPI);

            sint_simd_t imm2 = trunc_as_i(y);

            imm2 = plus(imm2, pi32_256_1);
            imm2 = bitwise_AND(imm2, pi32_256_inv1);
            y = floating<simd_type>(imm2);
            imm2 = minus(imm2, pi32_256_2);

            sint_simd_t imm0 = bitwise_ANDNOT(imm2, pi32_256_4);
            imm0 = shift_left<SHIFT_29>(imm0);

            imm2 = bitwise_AND(imm2, pi32_256_2);
            imm2 = equal(imm2, pi32_256_0).as_basic_simd<sint_simd_t>();

            simd_type sign_bit = reinterpret<simd_type>(imm0);
            simd_type poly_mask = reinterpret<simd_type>(imm2);

            x = fma(y, minus_cephes_DP1, x);
            x = fma(y, minus_cephes_DP2, x);
            x = fma(y, minus_cephes_DP3, x);

            y = coscof_p0;
            simd_type z = multiplies(x, x);

            y = fma(y, z, coscof_p1);
            y = fma(y, z, coscof_p2);
            y = multiplies(y, z);
            y = multiplies(y, z);
            simd_type tmp = multiplies(z, ps256_0p5);
            y = minus(y, tmp);
            y = plus(y, ps256_1);

            simd_type y2 = fma(sincof_p0, z, sincof_p1);
            y2 = fma(y2, z, sincof_p2);
            y2 = multiplies(y2, z);
            y2 = multiplies(y2, x);
            y2 = plus(y2, x);

            simd_type xmm3 = poly_mask;
            y2 = bitwise_AND(xmm3, y2);
            y = bitwise_ANDNOT(xmm3, y);
            y = plus(y, y2);

            y = bitwise_XOR(y, sign_bit);

            return y;
        }
    }

    float32x8 sin(float32x8 input) { return detail::sin_soft_simulation(input); }
    float64x4 sin(float64x4 input) { return detail::sin_soft_simulation(input); }
    float32x4 sin(float32x4 input) { return detail::sin_soft_simulation(input); }
    float64x2 sin(float64x2 input) { return detail::sin_soft_simulation(input); }

    float32x8 cos(float32x8 input) { return detail::cos_soft_simulation(input); }
    float64x4 cos(float64x4 input) { return detail::cos_soft_simulation(input); }
    float32x4 cos(float32x4 input) { return detail::cos_soft_simulation(input); }
    float64x2 cos(float64x2 input) { return detail::cos_soft_simulation(input); }
#endif
#if defined(_FOYE_SIMD_HAS_FP16_)
    float16x8 sin(float16x8 input)
    {
        float32x8 result32 = fyx::simd::sin(fyx::simd::expand<float32x8>(input));
        return fyx::simd::narrowing<float16x8>(result32);
    }

    float16x16 sin(float16x16 input)
    {
        float32x8 result_low = fyx::simd::sin(fyx::simd::expand_low<float32x8>(input));
        float32x8 result_high = fyx::simd::sin(fyx::simd::expand_high<float32x8>(input));
        return fyx::simd::merge(
            fyx::simd::narrowing<float16x8>(result_low),
            fyx::simd::narrowing<float16x8>(result_high)
        );
    }

    float16x8 cos(float16x8 input)
    {
        float32x8 result32 = fyx::simd::cos(fyx::simd::expand<float32x8>(input));
        return fyx::simd::narrowing<float16x8>(result32);
    }

    float16x16 cos(float16x16 input)
    {
        float32x8 result_low = fyx::simd::cos(fyx::simd::expand_low<float32x8>(input));
        float32x8 result_high = fyx::simd::cos(fyx::simd::expand_high<float32x8>(input));
        return fyx::simd::merge(
            fyx::simd::narrowing<float16x8>(result_low),
            fyx::simd::narrowing<float16x8>(result_high)
        );
    }
#endif
#if defined(_FOYE_SIMD_HAS_BF16_)
    bfloat16x8 sin(bfloat16x8 input)
    {
        float32x8 result32 = fyx::simd::sin(fyx::simd::expand<float32x8>(input));
        return fyx::simd::narrowing<bfloat16x8>(result32);
    }

    bfloat16x16 sin(bfloat16x16 input)
    {
        float32x8 result_low = fyx::simd::sin(fyx::simd::expand_low<float32x8>(input));
        float32x8 result_high = fyx::simd::sin(fyx::simd::expand_high<float32x8>(input));
        return fyx::simd::merge(
            fyx::simd::narrowing<bfloat16x8>(result_low),
            fyx::simd::narrowing<bfloat16x8>(result_high)
        );
    }

    bfloat16x8 cos(bfloat16x8 input)
    {
        float32x8 result32 = fyx::simd::cos(fyx::simd::expand<float32x8>(input));
        return fyx::simd::narrowing<bfloat16x8>(result32);
    }

    bfloat16x16 cos(bfloat16x16 input)
    {
        float32x8 result_low = fyx::simd::cos(fyx::simd::expand_low<float32x8>(input));
        float32x8 result_high = fyx::simd::cos(fyx::simd::expand_high<float32x8>(input));
        return fyx::simd::merge(
            fyx::simd::narrowing<bfloat16x8>(result_low),
            fyx::simd::narrowing<bfloat16x8>(result_high)
        );
    }
#endif


#if defined(FOYE_SIMD_ENABLE_SVML)
    float32x8 asin(float32x8 input) { return float32x8{ _mm256_asin_ps(input.data) }; }
    float64x4 asin(float64x4 input) { return float64x4{ _mm256_asin_pd(input.data) }; }
    float32x4 asin(float32x4 input) { return float32x4{ _mm_asin_ps(input.data) }; }
    float64x2 asin(float64x2 input) { return float64x2{ _mm_asin_pd(input.data) }; }
#else
    namespace detail
    {
        template<typename simd_type>
        simd_type asinF32_soft_simulation(simd_type input)
        {
            using sint_simd_t = basic_simd<detail::integral_t<
                simd_type::scalar_bit_width, true>, simd_type::bit_width>;

            simd_type xx = input;
            const simd_type ABS_MASK = reinterpret<simd_type>(load_brocast<sint_simd_t>(0x7FFFFFFF));
            const simd_type ONE = load_brocast<simd_type>(1.0f);
            const simd_type ZERO = allzero_bits_as<simd_type>();
            const simd_type PI_OVER_2 = load_brocast<simd_type>(1.57079632f);
            const simd_type TWO = load_brocast<simd_type>(2.0f);
            const simd_type SMALL_THRESHOLD = load_brocast<simd_type>(1.0e-4f);

            const simd_type ASIN_COEFF0 = load_brocast<simd_type>(-0.0187293f);
            const simd_type ASIN_COEFF1 = load_brocast<simd_type>(0.0742610f);
            const simd_type ASIN_COEFF2 = load_brocast<simd_type>(-0.2121144f);
            const simd_type ASIN_COEFF3 = load_brocast<simd_type>(1.5707288f);

            simd_type abs_x = bitwise_AND(xx, ABS_MASK);

            mask_from_simd_t<simd_type> output_nan = greater(abs_x, ONE);
            mask_from_simd_t<simd_type> small_value = less(abs_x, SMALL_THRESHOLD);

            simd_type negate = where_assign(ZERO, ONE, less(xx, ZERO));
            simd_type x = abs_x;

            simd_type result = fma(x, ASIN_COEFF0, ASIN_COEFF1);
            result = fma(x, result, ASIN_COEFF2);
            result = fma(x, result, ASIN_COEFF3);

            simd_type sqrt_term = sqrt(minus(ONE, x));
            result = fnma(sqrt_term, result, PI_OVER_2);

            simd_type two_times_result = multiplies(TWO, result);
            result = fnma(two_times_result, negate, result);

            result = bitwise_OR(result, output_nan.as_basic_simd<simd_type>());
            result = where_assign(result, xx, small_value);
            return result;
        }
    }

    float32x8 asin(float32x8 input) { return detail::asinF32_soft_simulation(input); }
    //float64x4 asin(float64x4 input) { return detail::asinF64_soft_simulation(input); }
    float32x4 asin(float32x4 input) { return detail::asinF32_soft_simulation(input); }
    //float64x2 asin(float64x2 input) { return detail::asinF64_soft_simulation(input); }
#endif
#if defined(_FOYE_SIMD_HAS_FP16_)
    float16x8 asin(float16x8 input)
    {
        float32x8 result32 = fyx::simd::asin(fyx::simd::expand<float32x8>(input));
        return fyx::simd::narrowing<float16x8>(result32);
    }

    float16x16 asin(float16x16 input)
    {
        float32x8 result_low = fyx::simd::asin(fyx::simd::expand_low<float32x8>(input));
        float32x8 result_high = fyx::simd::asin(fyx::simd::expand_high<float32x8>(input));
        return fyx::simd::merge(
            fyx::simd::narrowing<float16x8>(result_low),
            fyx::simd::narrowing<float16x8>(result_high)
        );
    }
#endif
#if defined(_FOYE_SIMD_HAS_BF16_)
    bfloat16x8 asin(bfloat16x8 input)
    {
        float32x8 result32 = fyx::simd::asin(fyx::simd::expand<float32x8>(input));
        return fyx::simd::narrowing<bfloat16x8>(result32);
    }

    bfloat16x16 asin(bfloat16x16 input)
    {
        float32x8 result_low = fyx::simd::asin(fyx::simd::expand_low<float32x8>(input));
        float32x8 result_high = fyx::simd::asin(fyx::simd::expand_high<float32x8>(input));
        return fyx::simd::merge(
            fyx::simd::narrowing<bfloat16x8>(result_low),
            fyx::simd::narrowing<bfloat16x8>(result_high)
        );
    }
#endif

#if defined(FOYE_SIMD_ENABLE_SVML)
    float32x8 tan(float32x8 input) { return float32x8{ _mm256_tan_ps(input.data) }; }
    float64x4 tan(float64x4 input) { return float64x4{ _mm256_tan_pd(input.data) }; }
    float32x4 tan(float32x4 input) { return float32x4{ _mm_tan_ps(input.data) }; }
    float64x2 tan(float64x2 input) { return float64x2{ _mm_tan_pd(input.data) }; }
#else
#endif


}

#endif
