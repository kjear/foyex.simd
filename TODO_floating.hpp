#ifndef _FOYE_SIMDXTODO_HPP_
#define _FOYE_SIMDXTODO_HPP_
#pragma once

#include "simd_def.hpp"

namespace fyx::simd::test
{

#if defined(FOYE_SIMD_ENABLE_SVML)
    float32x8 cdfnorminv(float32x8 input) { return float32x8{ _mm256_cdfnorminv_ps(input.data) }; }
    float64x4 cdfnorminv(float64x4 input) { return float64x4{ _mm256_cdfnorminv_pd(input.data) }; }
    float32x4 cdfnorminv(float32x4 input) { return float32x4{ _mm_cdfnorminv_ps(input.data) }; }
    float64x2 cdfnorminv(float64x2 input) { return float64x2{ _mm_cdfnorminv_pd(input.data) }; }
#else
#endif

#if defined(FOYE_SIMD_ENABLE_SVML)
    float32x8 erfinv(float32x8 input) { return float32x8{ _mm256_erfinv_ps(input.data) }; }
    float64x4 erfinv(float64x4 input) { return float64x4{ _mm256_erfinv_pd(input.data) }; }
    float32x4 erfinv(float32x4 input) { return float32x4{ _mm_erfinv_ps(input.data) }; }
    float64x2 erfinv(float64x2 input) { return float64x2{ _mm_erfinv_pd(input.data) }; }
#else
#endif

#if defined(FOYE_SIMD_ENABLE_SVML)
    float32x8 exp2(float32x8 input) { return float32x8{ _mm256_exp2_ps(input.data) }; }
    float64x4 exp2(float64x4 input) { return float64x4{ _mm256_exp2_pd(input.data) }; }
    float32x4 exp2(float32x4 input) { return float32x4{ _mm_exp2_ps(input.data) }; }
    float64x2 exp2(float64x2 input) { return float64x2{ _mm_exp2_pd(input.data) }; }
#else
#endif

#if defined(FOYE_SIMD_ENABLE_SVML)
    float32x8 exp10(float32x8 input) { return float32x8{ _mm256_exp10_ps(input.data) }; }
    float64x4 exp10(float64x4 input) { return float64x4{ _mm256_exp10_pd(input.data) }; }
    float32x4 exp10(float32x4 input) { return float32x4{ _mm_exp10_ps(input.data) }; }
    float64x2 exp10(float64x2 input) { return float64x2{ _mm_exp10_pd(input.data) }; }
#else
#endif

#if defined(FOYE_SIMD_ENABLE_SVML)
    float32x8 log2(float32x8 input) { return float32x8{ _mm256_log2_ps(input.data) }; }
    float64x4 log2(float64x4 input) { return float64x4{ _mm256_log2_pd(input.data) }; }
    float32x4 log2(float32x4 input) { return float32x4{ _mm_log2_ps(input.data) }; }
    float64x2 log2(float64x2 input) { return float64x2{ _mm_log2_pd(input.data) }; }
#else
#endif

#if defined(FOYE_SIMD_ENABLE_SVML)
    float32x8 log10(float32x8 input) { return float32x8{ _mm256_log10_ps(input.data) }; }
    float64x4 log10(float64x4 input) { return float64x4{ _mm256_log10_pd(input.data) }; }
    float32x4 log10(float32x4 input) { return float32x4{ _mm_log10_ps(input.data) }; }
    float64x2 log10(float64x2 input) { return float64x2{ _mm_log10_pd(input.data) }; }
#else
#endif

#if defined(FOYE_SIMD_ENABLE_SVML)
    float32x8 sin(float32x8 input) { return float32x8{ _mm256_sin_ps(input.data) }; }
    float64x4 sin(float64x4 input) { return float64x4{ _mm256_sin_pd(input.data) }; }
    float32x4 sin(float32x4 input) { return float32x4{ _mm_sin_ps(input.data) }; }
    float64x2 sin(float64x2 input) { return float64x2{ _mm_sin_pd(input.data) }; }
#else
#endif

#if defined(FOYE_SIMD_ENABLE_SVML)
    float32x8 asin(float32x8 input) { return float32x8{ _mm256_asin_ps(input.data) }; }
    float64x4 asin(float64x4 input) { return float64x4{ _mm256_asin_pd(input.data) }; }
    float32x4 asin(float32x4 input) { return float32x4{ _mm_asin_ps(input.data) }; }
    float64x2 asin(float64x2 input) { return float64x2{ _mm_asin_pd(input.data) }; }
#else
#endif

#if defined(FOYE_SIMD_ENABLE_SVML)
    float32x8 cos(float32x8 input) { return float32x8{ _mm256_cos_ps(input.data) }; }
    float64x4 cos(float64x4 input) { return float64x4{ _mm256_cos_pd(input.data) }; }
    float32x4 cos(float32x4 input) { return float32x4{ _mm_cos_ps(input.data) }; }
    float64x2 cos(float64x2 input) { return float64x2{ _mm_cos_pd(input.data) }; }
#else
#endif

#if defined(FOYE_SIMD_ENABLE_SVML)
    float32x8 acos(float32x8 input) { return float32x8{ _mm256_acos_ps(input.data) }; }
    float64x4 acos(float64x4 input) { return float64x4{ _mm256_acos_pd(input.data) }; }
    float32x4 acos(float32x4 input) { return float32x4{ _mm_acos_ps(input.data) }; }
    float64x2 acos(float64x2 input) { return float64x2{ _mm_acos_pd(input.data) }; }
#endif

#if defined(FOYE_SIMD_ENABLE_SVML)
    float32x8 tan(float32x8 input) { return float32x8{ _mm256_tan_ps(input.data) }; }
    float64x4 tan(float64x4 input) { return float64x4{ _mm256_tan_pd(input.data) }; }
    float32x4 tan(float32x4 input) { return float32x4{ _mm_tan_ps(input.data) }; }
    float64x2 tan(float64x2 input) { return float64x2{ _mm_tan_pd(input.data) }; }
#else
#endif

#if defined(FOYE_SIMD_ENABLE_SVML)
    float32x8 atan(float32x8 input) { return float32x8{ _mm256_atan_ps(input.data) }; }
    float64x4 atan(float64x4 input) { return float64x4{ _mm256_atan_pd(input.data) }; }
    float32x4 atan(float32x4 input) { return float32x4{ _mm_atan_ps(input.data) }; }
    float64x2 atan(float64x2 input) { return float64x2{ _mm_atan_pd(input.data) }; }
#else
#endif

#if defined(FOYE_SIMD_ENABLE_SVML)
    float32x8 sind(float32x8 input) { return float32x8{ _mm256_sind_ps(input.data) }; }
    float64x4 sind(float64x4 input) { return float64x4{ _mm256_sind_pd(input.data) }; }
    float32x4 sind(float32x4 input) { return float32x4{ _mm_sind_ps(input.data) }; }
    float64x2 sind(float64x2 input) { return float64x2{ _mm_sind_pd(input.data) }; }
#else
#endif

#if defined(FOYE_SIMD_ENABLE_SVML)
    float32x8 cosd(float32x8 input) { return float32x8{ _mm256_cosd_ps(input.data) }; }
    float64x4 cosd(float64x4 input) { return float64x4{ _mm256_cosd_pd(input.data) }; }
    float32x4 cosd(float32x4 input) { return float32x4{ _mm_cosd_ps(input.data) }; }
    float64x2 cosd(float64x2 input) { return float64x2{ _mm_cosd_pd(input.data) }; }
#else
#endif

#if defined(FOYE_SIMD_ENABLE_SVML)
    float32x8 tand(float32x8 input) { return float32x8{ _mm256_tand_ps(input.data) }; }
    float64x4 tand(float64x4 input) { return float64x4{ _mm256_tand_pd(input.data) }; }
    float32x4 tand(float32x4 input) { return float32x4{ _mm_tand_ps(input.data) }; }
    float64x2 tand(float64x2 input) { return float64x2{ _mm_tand_pd(input.data) }; }
#else
#endif

#if defined(FOYE_SIMD_ENABLE_SVML)
    float32x8 sinh(float32x8 input) { return float32x8{ _mm256_sinh_ps(input.data) }; }
    float64x4 sinh(float64x4 input) { return float64x4{ _mm256_sinh_pd(input.data) }; }
    float32x4 sinh(float32x4 input) { return float32x4{ _mm_sinh_ps(input.data) }; }
    float64x2 sinh(float64x2 input) { return float64x2{ _mm_sinh_pd(input.data) }; }
#else
#endif

#if defined(FOYE_SIMD_ENABLE_SVML)
    float32x8 cosh(float32x8 input) { return float32x8{ _mm256_cosh_ps(input.data) }; }
    float64x4 cosh(float64x4 input) { return float64x4{ _mm256_cosh_pd(input.data) }; }
    float32x4 cosh(float32x4 input) { return float32x4{ _mm_cosh_ps(input.data) }; }
    float64x2 cosh(float64x2 input) { return float64x2{ _mm_cosh_pd(input.data) }; }
#else
#endif

#if defined(FOYE_SIMD_ENABLE_SVML)
    float32x8 tanh(float32x8 input) { return float32x8{ _mm256_tanh_ps(input.data) }; }
    float64x4 tanh(float64x4 input) { return float64x4{ _mm256_tanh_pd(input.data) }; }
    float32x4 tanh(float32x4 input) { return float32x4{ _mm_tanh_ps(input.data) }; }
    float64x2 tanh(float64x2 input) { return float64x2{ _mm_tanh_pd(input.data) }; }
#else
#endif

#if defined(FOYE_SIMD_ENABLE_SVML)
    float32x8 asinh(float32x8 input) { return float32x8{ _mm256_asinh_ps(input.data) }; }
    float64x4 asinh(float64x4 input) { return float64x4{ _mm256_asinh_pd(input.data) }; }
    float32x4 asinh(float32x4 input) { return float32x4{ _mm_asinh_ps(input.data) }; }
    float64x2 asinh(float64x2 input) { return float64x2{ _mm_asinh_pd(input.data) }; }
#else
#endif

#if defined(FOYE_SIMD_ENABLE_SVML)
    float32x8 acosh(float32x8 input) { return float32x8{ _mm256_acosh_ps(input.data) }; }
    float64x4 acosh(float64x4 input) { return float64x4{ _mm256_acosh_pd(input.data) }; }
    float32x4 acosh(float32x4 input) { return float32x4{ _mm_acosh_ps(input.data) }; }
    float64x2 acosh(float64x2 input) { return float64x2{ _mm_acosh_pd(input.data) }; }
#else
#endif

#if defined(FOYE_SIMD_ENABLE_SVML)
    float32x8 atanh(float32x8 input) { return float32x8{ _mm256_atanh_ps(input.data) }; }
    float64x4 atanh(float64x4 input) { return float64x4{ _mm256_atanh_pd(input.data) }; }
    float32x4 atanh(float32x4 input) { return float32x4{ _mm_atanh_ps(input.data) }; }
    float64x2 atanh(float64x2 input) { return float64x2{ _mm_atanh_pd(input.data) }; }
#else
#endif

#if defined(FOYE_SIMD_ENABLE_SVML)
    float32x8 logb(float32x8 input) { return float32x8{ _mm256_logb_ps(input.data) }; }
    float64x4 logb(float64x4 input) { return float64x4{ _mm256_logb_pd(input.data) }; }
    float32x4 logb(float32x4 input) { return float32x4{ _mm_logb_ps(input.data) }; }
    float64x2 logb(float64x2 input) { return float64x2{ _mm_logb_pd(input.data) }; }
#else
#endif

#if defined(FOYE_SIMD_ENABLE_SVML)
    float32x8 expm1(float32x8 input) { return float32x8{ _mm256_expm1_ps(input.data) }; }
    float64x4 expm1(float64x4 input) { return float64x4{ _mm256_expm1_pd(input.data) }; }
    float32x4 expm1(float32x4 input) { return float32x4{ _mm_expm1_ps(input.data) }; }
    float64x2 expm1(float64x2 input) { return float64x2{ _mm_expm1_pd(input.data) }; }
#else
#endif

#if defined(FOYE_SIMD_ENABLE_SVML)
    float32x8 invsqrt(float32x8 input) { return float32x8{ _mm256_invsqrt_ps(input.data) }; }
    float64x4 invsqrt(float64x4 input) { return float64x4{ _mm256_invsqrt_pd(input.data) }; }
    float32x4 invsqrt(float32x4 input) { return float32x4{ _mm_invsqrt_ps(input.data) }; }
    float64x2 invsqrt(float64x2 input) { return float64x2{ _mm_invsqrt_pd(input.data) }; }
#else
#endif

#if defined(FOYE_SIMD_ENABLE_SVML)
    float32x8 invcbrt(float32x8 input) { return float32x8{ _mm256_invcbrt_ps(input.data) }; }
    float64x4 invcbrt(float64x4 input) { return float64x4{ _mm256_invcbrt_pd(input.data) }; }
    float32x4 invcbrt(float32x4 input) { return float32x4{ _mm_invcbrt_ps(input.data) }; }
    float64x2 invcbrt(float64x2 input) { return float64x2{ _mm_invcbrt_pd(input.data) }; }
#else
#endif

#if defined(FOYE_SIMD_ENABLE_SVML)
    float32x8 atan2(float32x8 arg0, float32x8 arg1) { return float32x8{ _mm256_atan2_ps(arg0.data, arg1.data) }; }
    float64x4 atan2(float64x4 arg0, float64x4 arg1) { return float64x4{ _mm256_atan2_pd(arg0.data, arg1.data) }; }
    float32x4 atan2(float32x4 arg0, float32x4 arg1) { return float32x4{ _mm_atan2_ps(arg0.data, arg1.data) }; }
    float64x2 atan2(float64x2 arg0, float64x2 arg1) { return float64x2{ _mm_atan2_pd(arg0.data, arg1.data) }; }
#else
#endif
}

#endif