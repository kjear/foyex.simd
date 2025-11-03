#ifndef _FOYE_SIMD_UTILITY_HPP_
#define _FOYE_SIMD_UTILITY_HPP_
#pragma once

#include <immintrin.h>
#include <intrin.h>
#include <type_traits>
#include <bit>
#include <cstdint>
#include <utility>
#include <format>
#include <iostream>
#include <cmath>
#include <limits>
#include <cfenv>
#include <array>

#if __has_include("foye_float16.hpp")
#include "foye_float16.hpp"
#define _FOYE_SIMD_HAS_FP16_
#endif

#if __has_include("foye_bfloat16.hpp")
#include "foye_bfloat16.hpp"
#define _FOYE_SIMD_HAS_BF16_
#endif

#define FOYE_SIMD_ERROR_WHEN_CALLED(text) __declspec(deprecated(text))

#if !defined(FOYE_SIMD_DISABLE_PERFORMANCE_NOTICE)
#pragma warning(error: 4996)
#define FOYE_SIMD_PERFORMANCE_MATTER \
	FOYE_SIMD_ERROR_WHEN_CALLED("The design purpose of this function is to prioritize convenience. "\
						         "If performance is to be considered, please use other solutions. "\
                                 "or define FOYE_SIMD_DISABLE_PERFORMANCE_NOTICE to disable this error")
#else
#define FOYE_SIMD_PERFORMANCE_MATTER
#endif

namespace fyx::simd::detail
{
    template<typename...>
    struct dependent_false : std::false_type {};
}

#define DEF_NOTSUPPORTED_IMPLEMENT(funcd) \
template<typename T = void> \
funcd \
{ \
    static_assert(fyx::simd::detail::dependent_false<T>::value, \
        "There is no instruction that supports this operation to implement function: " #funcd \
        ". you define _FOYE_ENABLE_EMULATED_ to use a simulated version of this function"); \
}

#define DEF_NOSUITABLE_IMPLEMENT(funcd) \
template<typename T = void> \
funcd \
{ \
    static_assert(fyx::simd::detail::dependent_false<T>::value, \
        "There is no suitable instruction combination to achieve this function: " #funcd); \
}

#define FOYE_SIMD_UNIMPLEMENTED static_assert(sizeof(int) == 0, "This function is not implemented yet")


#define FOYE_SIMD_DEBUG_MSG(msg) \
do\
{\
    std::cout << msg << std::endl;\
} while (0)


namespace fyx::simd
{
    template<typename T>
    constexpr bool is_available_scalar_type_for_basic_simd = (
        std::is_same_v<T, std::uint8_t> ||
        std::is_same_v<T, std::uint16_t> ||
        std::is_same_v<T, std::uint32_t> ||
        std::is_same_v<T, std::uint64_t> ||
        std::is_same_v<T, std::int8_t> ||
        std::is_same_v<T, std::int16_t> ||
        std::is_same_v<T, std::int32_t> ||
        std::is_same_v<T, std::int64_t> ||
        std::is_same_v<T, float> ||
        std::is_same_v<T, double>
#if defined(_FOYE_SIMD_HAS_FP16_)
        || std::is_same_v<T, fy::float16>
#endif
#if defined(_FOYE_SIMD_HAS_BF16_)
        || std::is_same_v<T, fy::bfloat16>
#endif
        );

#define FOYE_SIMD_CHECK_SCALAR_AVAILABLE(_scalar_type_) \
    static_assert(fyx::simd::is_available_scalar_type_for_basic_simd<_scalar_type_>, \
    "Scalar type " #_scalar_type_ " is not available for basic_simd")
}

#if defined(_FOYE_SIMD_HAS_FP16_) || defined(_FOYE_SIMD_HAS_BF16_)
namespace fyx::simd
{
#ifdef _FOYE_SIMD_HAS_FP16_
#define FOYE_SIMD_CMPCHECK_IS_FP16(t_tocheck) constexpr (std::is_same_v<t_tocheck, fy::float16>)
#define cvt8lane_fp32_to_fp16(vec) (_mm256_cvtps_ph(vec, _MM_FROUND_CUR_DIRECTION))
#define cvt8lane_fp16_to_fp32(vec) (_mm256_cvtph_ps(vec))
#define dispatch_8lane_fp32intrin_tofp16(vec, expr) (cvt8lane_fp32_to_fp16(expr(cvt8lane_fp16_to_fp32(vec))))
#define dispatch_16lane_fp32intrin_tofp16(vec, expr) (FOYE_SIMD_MERGE_i(\
        cvt8lane_fp32_to_fp16(expr(cvt8lane_fp16_to_fp32(FOYE_SIMD_EXTRACT_LOW_i(input.data)))),\
        cvt8lane_fp32_to_fp16(expr(cvt8lane_fp16_to_fp32(FOYE_SIMD_EXTRACT_HIGH_i(input.data))))))
#else
#define cvt8lane_fp16_to_fp32(...)
#define cvt8lane_fp32_to_fp16(...)
#define dispatch_16lane_fp32intrin_tofp16(...)
#define dispatch_8lane_fp32intrin_tofp16(...)
#endif
#ifdef _FOYE_SIMD_HAS_BF16_
#define FOYE_SIMD_CMPCHECK_IS_BF16(t_tocheck) constexpr (std::is_same_v<t_tocheck, fy::bfloat16>)

#define cvt8lane_bf16_to_fp32(input) (_mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(input), 16)))

inline __m128i cvt8lane_fp32_to_bf16___(__m256 input)
{
    const __m256i v_exp_mask = _mm256_set1_epi32(0x7F800000);
    const __m256i v_mant_mask = _mm256_set1_epi32(0x007FFFFF);
    const __m256i v_zero = _mm256_setzero_si256();

    const __m256i v = _mm256_castps_si256(input);

    const __m256i is_nan = _mm256_and_si256(
        _mm256_cmpeq_epi32(_mm256_and_si256(v, v_exp_mask), v_exp_mask),
        _mm256_cmpgt_epi32(_mm256_and_si256(v, v_mant_mask), v_zero));

    __m256i shifted = _mm256_srli_epi32(v, 16);
    __m256i shifted_low = _mm256_and_si256(shifted, _mm256_set1_epi32(0x0000FFFF));

    __m256i shifted_updata = _mm256_or_si256(shifted,
        _mm256_and_si256(
            _mm256_set1_epi32(0x00000001),
            _mm256_and_si256(is_nan, _mm256_cmpeq_epi32(shifted_low, v_zero))));

    return _mm_packus_epi32(
        _mm256_extractf128_si256(shifted_updata, 0),
        _mm256_extractf128_si256(shifted_updata, 1));
}
#define cvt8lane_fp32_to_bf16(input) (fyx::simd::cvt8lane_fp32_to_bf16___(input))
#define dispatch_8lane_fp32intrin_tobf16(vec, expr) (cvt8lane_fp32_to_bf16(expr(cvt8lane_bf16_to_fp32(vec))))
#define dispatch_16lane_fp32intrin_tobf16(vec, expr) (FOYE_SIMD_MERGE_i(\
        cvt8lane_fp32_to_bf16(expr(cvt8lane_bf16_to_fp32(FOYE_SIMD_EXTRACT_LOW_i(input.data)))),\
        cvt8lane_fp32_to_bf16(expr(cvt8lane_bf16_to_fp32(FOYE_SIMD_EXTRACT_HIGH_i(input.data))))))
#else
#define cvt8lane_fp32_to_bf16(...)
#define cvt8lane_bf16_to_fp32(...)
#define dispatch_8lane_fp32intrin_tobf16(...)
#define dispatch_16lane_fp32intrin_tobf16(...)
#endif
}
#endif

#define _FOYE_SIMD_CMPLT_PS_(lhs, rhs) (_mm256_cmp_ps(lhs, rhs, _CMP_LT_OQ))
#define _FOYE_SIMD_CMPGT_PS_(lhs, rhs) (_mm256_cmp_ps(lhs, rhs, _CMP_GT_OQ))
#define _FOYE_SIMD_CMPEQ_PS_(lhs, rhs) (_mm256_cmp_ps(lhs, rhs, _CMP_EQ_OQ))
#define _FOYE_SIMD_CMPORD_PS_(lhs, rhs) (_mm256_cmp_ps(lhs, rhs, _CMP_ORD_Q))
#define _FOYE_SIMD_ABS_PS_(arg) (_mm256_andnot_ps(_mm256_set1_ps(-0.0f), arg))


#if defined(_FOYE_SIMD_HAS_FP16_) || defined(_FOYE_SIMD_HAS_BF16_)
#define _FOYE_SIMD_DISPATCH_8LANE_HALF_V_VV(lhs, rhs, func, to_half, from_half) (to_half(func(from_half(lhs), from_half(rhs))))
#define _FOYE_SIMD_DISPATCH_16LANE_HALF_V_VV(lhs, rhs, func, to_half, from_half) \
    (_mm256_inserti128_si256(_mm256_castsi128_si256( \
        to_half(func(from_half(_mm256_castsi256_si128(lhs)), \
                                   from_half(_mm256_castsi256_si128(rhs))))),\
        to_half(func(from_half(_mm256_extracti128_si256(lhs, 1)), \
                                   from_half(_mm256_extracti128_si256(rhs, 1)))), 0x1))

#define _FOYE_SIMD_DISPATCH_8LANE_HALF_V_V(input, func, to_half, from_half) (to_half(func(from_half(input))))
#define _FOYE_SIMD_DISPATCH_16LANE_HALF_V_V(input, func, to_half, from_half) \
    (_mm256_inserti128_si256(_mm256_castsi128_si256( \
    to_half(func(from_half(_mm256_castsi256_si128(input))))),\
    to_half(func(from_half(_mm256_extracti128_si256(input, 1)))), 0x1))


#define _FOYE_SIMD_DISPATCH_8LANE_HALF_V_V(input, func, to_half, from_half) (to_half(func(from_half(input))))
#define _FOYE_SIMD_DISPATCH_16LANE_HALF_V_V(input, func, to_half, from_half) \
    (_mm256_inserti128_si256(_mm256_castsi128_si256( \
    to_half(func(from_half(_mm256_castsi256_si128(input))))),\
    to_half(func(from_half(_mm256_extracti128_si256(input, 1)))), 0x1))

#ifdef _FOYE_SIMD_HAS_FP16_
#define _FOYE_SIMD_DISPATCH_8LANE_FP16_V_VV(lhs, rhs, func) (_FOYE_SIMD_DISPATCH_8LANE_HALF_V_VV(lhs, rhs, func, cvt8lane_fp32_to_fp16, cvt8lane_fp16_to_fp32))
#define _FOYE_SIMD_DISPATCH_16LANE_FP16_V_VV(lhs, rhs, func) (_FOYE_SIMD_DISPATCH_16LANE_HALF_V_VV(lhs, rhs, func, cvt8lane_fp32_to_fp16, cvt8lane_fp16_to_fp32))
#define _FOYE_SIMD_DISPATCH_8LANE_FP16_V_V(input, func) (_FOYE_SIMD_DISPATCH_8LANE_HALF_V_V(input, func, cvt8lane_fp32_to_fp16, cvt8lane_fp16_to_fp32))
#define _FOYE_SIMD_DISPATCH_16LANE_FP16_V_V(input, func) (_FOYE_SIMD_DISPATCH_16LANE_HALF_V_V(input, func, cvt8lane_fp32_to_fp16, cvt8lane_fp16_to_fp32))

#define _FOYE_SIMD_WARP_FP16_V_VV(funcname, func)\
float16x8 funcname(float16x8 lhs, float16x8 rhs) { return float16x8{ _FOYE_SIMD_DISPATCH_8LANE_FP16_V_VV(lhs.data, rhs.data, func) }; } \
float16x16 funcname(float16x16 lhs, float16x16 rhs) { return float16x16{ _FOYE_SIMD_DISPATCH_16LANE_FP16_V_VV(lhs.data, rhs.data, func) }; }

#define _FOYE_SIMD_WARP_FP16_V_V(funcname, func)\
float16x8 funcname(float16x8 input) { return float16x8{ _FOYE_SIMD_DISPATCH_8LANE_FP16_V_V(input.data, func)} ; }\
float16x16 funcname(float16x16 input) { return float16x16{ _FOYE_SIMD_DISPATCH_16LANE_FP16_V_V(input.data, func) }; }

#else
#define _FOYE_SIMD_DISPATCH_8LANE_FP16_V_VV(...)
#define _FOYE_SIMD_DISPATCH_16LANE_FP16_V_VV(...)
#define _FOYE_SIMD_DISPATCH_8LANE_FP16_V_V(...)
#define _FOYE_SIMD_DISPATCH_16LANE_FP16_V_V(...)
#define _FOYE_SIMD_WARP_FP16_V_VV(...)
#define _FOYE_SIMD_WARP_FP16_V_V(...)
#endif

#ifdef _FOYE_SIMD_HAS_BF16_
#define _FOYE_SIMD_DISPATCH_8LANE_BF16_V_VV(lhs, rhs, func) (_FOYE_SIMD_DISPATCH_8LANE_HALF_V_VV(lhs, rhs, func, cvt8lane_fp32_to_bf16, cvt8lane_bf16_to_fp32))
#define _FOYE_SIMD_DISPATCH_16LANE_BF16_V_VV(lhs, rhs, func) (_FOYE_SIMD_DISPATCH_16LANE_HALF_V_VV(lhs, rhs, func, cvt8lane_fp32_to_bf16, cvt8lane_bf16_to_fp32))
#define _FOYE_SIMD_DISPATCH_8LANE_BF16_V_V(input, func) (_FOYE_SIMD_DISPATCH_8LANE_HALF_V_V(input, func, cvt8lane_fp32_to_bf16, cvt8lane_bf16_to_fp32))
#define _FOYE_SIMD_DISPATCH_16LANE_BF16_V_V(input, func) (_FOYE_SIMD_DISPATCH_16LANE_HALF_V_V(input, func, cvt8lane_fp32_to_bf16, cvt8lane_bf16_to_fp32))

#define _FOYE_SIMD_WARP_BF16_V_VV(funcname, func)\
bfloat16x8 funcname(bfloat16x8 lhs, bfloat16x8 rhs) { return bfloat16x8{ _FOYE_SIMD_DISPATCH_8LANE_BF16_V_VV(lhs.data, rhs.data, func) }; } \
bfloat16x16 funcname(bfloat16x16 lhs, bfloat16x16 rhs) { return bfloat16x16{ _FOYE_SIMD_DISPATCH_16LANE_BF16_V_VV(lhs.data, rhs.data, func) }; }

#define _FOYE_SIMD_WARP_BF16_V_V(funcname, func)\
bfloat16x8 funcname(bfloat16x8 input) { return bfloat16x8{ _FOYE_SIMD_DISPATCH_8LANE_BF16_V_V(input.data, func) }; }\
bfloat16x16 funcname(bfloat16x16 input) { return bfloat16x16{ _FOYE_SIMD_DISPATCH_16LANE_BF16_V_V(input.data, func) }; }

#else
#define _FOYE_SIMD_DISPATCH_8LANE_BF16_V_VV(...)
#define _FOYE_SIMD_DISPATCH_16LANE_BF16_V_VV(...)
#define _FOYE_SIMD_DISPATCH_8LANE_BF16_V_V(...)
#define _FOYE_SIMD_DISPATCH_16LANE_BF16_V_V(...)
#define _FOYE_SIMD_WARP_BF16_V_VV(...)
#define _FOYE_SIMD_WARP_BF16_V_V(...)
#endif

#define _FOYE_SIMD_DISPATCH_8LANE_V_VV_(funcname, func) _FOYE_SIMD_WARP_FP16_V_VV(funcname, func) _FOYE_SIMD_WARP_BF16_V_VV(funcname, func)
#define _FOYE_SIMD_DISPATCH_8LANE_V_V_(funcname, func) _FOYE_SIMD_WARP_FP16_V_V(funcname, func) _FOYE_SIMD_WARP_BF16_V_V(funcname, func)
#else
#define _FOYE_SIMD_DISPATCH_8LANE_V_VV_(...)
#define _FOYE_SIMD_DISPATCH_8LANE_V_V_(...)
#endif

namespace fyx::simd::detail
{
    template<std::size_t bits_width, bool has_sign_bit>
    struct integral;

    template<> struct integral<8, true> { using type = std::int8_t; };
    template<> struct integral<16, true> { using type = std::int16_t; };
    template<> struct integral<32, true> { using type = std::int32_t; };
    template<> struct integral<64, true> { using type = std::int64_t; };

    template<> struct integral<8, false> { using type = std::uint8_t; };
    template<> struct integral<16, false> { using type = std::uint16_t; };
    template<> struct integral<32, false> { using type = std::uint32_t; };
    template<> struct integral<64, false> { using type = std::uint64_t; };

    template<std::size_t bits_width, bool has_sign_bit>
    using integral_t = typename integral<bits_width, has_sign_bit>::type;
}

namespace fyx::simd::detail
{
    template<typename T>
    using vector_128_t = std::conditional_t<std::is_same_v<T, float>, __m128,
        std::conditional_t<std::is_same_v<T, double>, __m128d, __m128i>>;

    template<typename T>
    using vector_256_t = std::conditional_t<std::is_same_v<T, float>, __m256,
        std::conditional_t<std::is_same_v<T, double>, __m256d, __m256i>>;

    template<typename T>
    constexpr bool is_mm128_vector_type_v = (std::is_same_v<T, __m128>
        || std::is_same_v<T, __m128d> || std::is_same_v<T, __m128i>);

    template<typename T>
    constexpr bool is_mm256_vector_type_v = (std::is_same_v<T, __m256>
        || std::is_same_v<T, __m256d> || std::is_same_v<T, __m256i>);


    template<typename T>
    constexpr bool is_mm_vector_type_v = (fyx::simd::detail::is_mm128_vector_type_v<T> ||
        fyx::simd::detail::is_mm256_vector_type_v<T>);

    template<typename return_vector_t, std::size_t scalar_size, std::size_t lane_width> 
    struct setter_by_each_invoker
    {
        template<typename ... Args> 
        return_vector_t operator () (Args&& ... args)
        { return return_vector_t{}; }
    };

#define DEFINE_SETTER_INVOKER_SPECIALIZATION(return_type, scalar_size, lane_width, expr)\
template<> struct setter_by_each_invoker<return_type, scalar_size, lane_width>\
{\
    template<typename ... Args> return_type operator() (Args&& ... args)\
        requires(sizeof...(Args) == lane_width) { return expr(std::forward<Args>(args)...); }\
}
    DEFINE_SETTER_INVOKER_SPECIALIZATION(__m128i, 1, 16, _mm_setr_epi8);
    DEFINE_SETTER_INVOKER_SPECIALIZATION(__m128i, 2, 8, _mm_setr_epi16);
    DEFINE_SETTER_INVOKER_SPECIALIZATION(__m128i, 4, 4, _mm_setr_epi32);
    DEFINE_SETTER_INVOKER_SPECIALIZATION(__m128i, 8, 2, _mm_setr_epi64x);
    DEFINE_SETTER_INVOKER_SPECIALIZATION(__m128, 4, 4, _mm_setr_ps);
    DEFINE_SETTER_INVOKER_SPECIALIZATION(__m128d, 8, 2, _mm_setr_pd);

    DEFINE_SETTER_INVOKER_SPECIALIZATION(__m256i, 1, 32, _mm256_setr_epi8);
    DEFINE_SETTER_INVOKER_SPECIALIZATION(__m256i, 2, 16, _mm256_setr_epi16);
    DEFINE_SETTER_INVOKER_SPECIALIZATION(__m256i, 4, 8, _mm256_setr_epi32);
    DEFINE_SETTER_INVOKER_SPECIALIZATION(__m256i, 8, 4, _mm256_setr_epi64x);
    DEFINE_SETTER_INVOKER_SPECIALIZATION(__m256, 4, 8, _mm256_setr_ps);
    DEFINE_SETTER_INVOKER_SPECIALIZATION(__m256d, 8, 4, _mm256_setr_pd);
#undef DEFINE_SETTER_INVOKER_SPECIALIZATION

    template<typename V> V zero_vec() { return V{}; }
    template<> __m128i zero_vec() { return _mm_setzero_si128(); }
    template<> __m128 zero_vec() { return _mm_setzero_ps(); }
    template<> __m128d zero_vec() { return _mm_setzero_pd(); }
    template<> __m256i zero_vec() { return _mm256_setzero_si256(); }
    template<> __m256 zero_vec() { return _mm256_setzero_ps(); }
    template<> __m256d zero_vec() { return _mm256_setzero_pd(); }

    template<typename T>
    consteval T all_ones_value() 
    {
        if constexpr (std::is_floating_point_v<T>) 
        {
            using IntType = std::conditional_t<sizeof(T) == sizeof(uint32_t), uint32_t, uint64_t>;
            IntType int_val = ~IntType{ 0 };
            return std::bit_cast<T>(int_val);
        }
        else 
        {
            return static_cast<T>(~static_cast<std::make_unsigned_t<T>>(0));
        }
    }

    template<typename T> T one_vec();
    template<> __m128i one_vec<__m128i>() { return _mm_set1_epi8(all_ones_value<char>()); }
    template<> __m128 one_vec<__m128>() { return _mm_set1_ps(all_ones_value<float>()); }
    template<> __m128d one_vec<__m128d>() { return _mm_set1_pd(all_ones_value<double>()); }
    template<> __m256i one_vec<__m256i>() { return _mm256_set1_epi8(all_ones_value<char>()); }
    template<> __m256 one_vec<__m256>() { return _mm256_set1_ps(all_ones_value<float>()); }
    template<> __m256d one_vec<__m256d>() { return _mm256_set1_pd(all_ones_value<double>()); }


    __m128i split_low(__m256i from) { return _mm256_castsi256_si128(from); }
    __m128 split_low(__m256 from) { return _mm256_castps256_ps128(from); }
    __m128d split_low(__m256d from) { return _mm256_castpd256_pd128(from); }
#define FOYE_SIMD_EXTRACT_LOW_i(from) (_mm256_castsi256_si128(from))
#define FOYE_SIMD_EXTRACT_LOW_f(from) (_mm256_castps256_ps128(from))
#define FOYE_SIMD_EXTRACT_LOW_d(from) (_mm256_castpd256_pd128(from))

    __m128i split_high(__m256i from) { return _mm256_extracti128_si256(from, 1); }
    __m128 split_high(__m256 from) { return _mm256_extractf128_ps(from, 1); }
    __m128d split_high(__m256d from) { return _mm256_extractf128_pd(from, 1); }
#define FOYE_SIMD_EXTRACT_HIGH_i(from) (_mm256_extracti128_si256(from, 1))
#define FOYE_SIMD_EXTRACT_HIGH_f(from) (_mm256_extractf128_ps(from, 1))
#define FOYE_SIMD_EXTRACT_HIGH_d(from) (_mm256_extractf128_pd(from, 1))

    __m256i merge(__m128i low, __m128i high) { return _mm256_inserti128_si256(_mm256_castsi128_si256(low), high, 0x1); }
    __m256 merge(__m128 low, __m128 high) { return _mm256_insertf128_ps(_mm256_castps128_ps256(low), high, 0x1); }
    __m256d merge(__m128d low, __m128d high) { return _mm256_insertf128_pd(_mm256_castpd128_pd256(low), high, 0x1); }
#define FOYE_SIMD_MERGE_i(low, high) (_mm256_inserti128_si256(_mm256_castsi128_si256(low), high, 0x1))
#define FOYE_SIMD_MERGE_f(low, high) (_mm256_insertf128_ps(_mm256_castps128_ps256(low), high, 0x1))
#define FOYE_SIMD_MERGE_d(low, high) (_mm256_insertf128_pd(_mm256_castpd128_pd256(low), high, 0x1))


    template<typename vector_type> vector_type load_aligned(const void*);
    template<> __m128i load_aligned<__m128i>(const void* mem_addr) { return _mm_load_si128(reinterpret_cast<const __m128i*>(mem_addr)); }
    template<> __m128 load_aligned<__m128>(const void* mem_addr) { return _mm_load_ps(reinterpret_cast<const float*>(mem_addr)); }
    template<> __m128d load_aligned<__m128d>(const void* mem_addr) { return _mm_load_pd(reinterpret_cast<const double*>(mem_addr)); }
    template<> __m256i load_aligned<__m256i>(const void* mem_addr) { return _mm256_load_si256(reinterpret_cast<const __m256i*>(mem_addr)); }
    template<> __m256 load_aligned<__m256>(const void* mem_addr) { return _mm256_load_ps(reinterpret_cast<const float*>(mem_addr)); }
    template<> __m256d load_aligned<__m256d>(const void* mem_addr) { return _mm256_load_pd(reinterpret_cast<const double*>(mem_addr)); }

    template<typename vector_type> vector_type load_unaligned(const void*);
    template<> __m128i load_unaligned<__m128i>(const void* mem_addr) { return _mm_loadu_si128(reinterpret_cast<const __m128i*>(mem_addr)); }
    template<> __m128 load_unaligned<__m128>(const void* mem_addr) { return _mm_loadu_ps(reinterpret_cast<const float*>(mem_addr)); }
    template<> __m128d load_unaligned<__m128d>(const void* mem_addr) { return _mm_loadu_pd(reinterpret_cast<const double*>(mem_addr)); }
    template<> __m256i load_unaligned<__m256i>(const void* mem_addr) { return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(mem_addr)); }
    template<> __m256 load_unaligned<__m256>(const void* mem_addr) { return _mm256_loadu_ps(reinterpret_cast<const float*>(mem_addr)); }
    template<> __m256d load_unaligned<__m256d>(const void* mem_addr) { return _mm256_loadu_pd(reinterpret_cast<const double*>(mem_addr)); }

    template<typename vector_type> void store_unaligned(vector_type, void*);
    template<> void store_unaligned<__m128i>(__m128i vec, void* mem_addr) { _mm_storeu_si128(reinterpret_cast<__m128i*>(mem_addr), vec); }
    template<> void store_unaligned<__m128>(__m128 vec, void* mem_addr) { _mm_storeu_ps(reinterpret_cast<float*>(mem_addr), vec); }
    template<> void store_unaligned<__m128d>(__m128d vec, void* mem_addr) { _mm_storeu_pd(reinterpret_cast<double*>(mem_addr), vec); }
    template<> void store_unaligned<__m256i>(__m256i vec, void* mem_addr) { _mm256_storeu_si256(reinterpret_cast<__m256i*>(mem_addr), vec); }
    template<> void store_unaligned<__m256>(__m256 vec, void* mem_addr) { _mm256_storeu_ps(reinterpret_cast<float*>(mem_addr), vec); }
    template<> void store_unaligned<__m256d>(__m256d vec, void* mem_addr) { _mm256_storeu_pd(reinterpret_cast<double*>(mem_addr), vec); }

    template<typename vector_type> void store_aligned(vector_type, void*);
    template<> void store_aligned<__m128i>(__m128i vec, void* mem_addr) { _mm_store_si128(reinterpret_cast<__m128i*>(mem_addr), vec); }
    template<> void store_aligned<__m128>(__m128 vec, void* mem_addr) { _mm_store_ps(reinterpret_cast<float*>(mem_addr), vec); }
    template<> void store_aligned<__m128d>(__m128d vec, void* mem_addr) { _mm_store_pd(reinterpret_cast<double*>(mem_addr), vec); }
    template<> void store_aligned<__m256i>(__m256i vec, void* mem_addr) { _mm256_store_si256(reinterpret_cast<__m256i*>(mem_addr), vec); }
    template<> void store_aligned<__m256>(__m256 vec, void* mem_addr) { _mm256_store_ps(reinterpret_cast<float*>(mem_addr), vec); }
    template<> void store_aligned<__m256d>(__m256d vec, void* mem_addr) { _mm256_store_pd(reinterpret_cast<double*>(mem_addr), vec); }

    template<typename vector_type> void store_stream(vector_type, void*);
    template<> void store_stream<__m128i>(__m128i vec, void* mem_addr) { _mm_stream_si128(reinterpret_cast<__m128i*>(mem_addr), vec); }
    template<> void store_stream<__m128>(__m128 vec, void* mem_addr) { _mm_stream_ps(reinterpret_cast<float*>(mem_addr), vec); }
    template<> void store_stream<__m128d>(__m128d vec, void* mem_addr) { _mm_stream_pd(reinterpret_cast<double*>(mem_addr), vec); }
    template<> void store_stream<__m256i>(__m256i vec, void* mem_addr) { _mm256_stream_si256(reinterpret_cast<__m256i*>(mem_addr), vec); }
    template<> void store_stream<__m256>(__m256 vec, void* mem_addr) { _mm256_stream_ps(reinterpret_cast<float*>(mem_addr), vec); }
    template<> void store_stream<__m256d>(__m256d vec, void* mem_addr) { _mm256_stream_pd(reinterpret_cast<double*>(mem_addr), vec); }

    template<typename return_type, typename scalar_type> return_type brocast(scalar_type) { return return_type{}; }
    template<> __m128i brocast<__m128i, std::uint8_t>(std::uint8_t scalar) { return _mm_set1_epi8(std::bit_cast<char>(scalar)); }
    template<> __m128i brocast<__m128i, std::uint16_t>(std::uint16_t scalar) { return _mm_set1_epi16(std::bit_cast<short>(scalar)); }
    template<> __m128i brocast<__m128i, std::uint32_t>(std::uint32_t scalar) { return _mm_set1_epi32(std::bit_cast<int>(scalar)); }
    template<> __m128i brocast<__m128i, std::uint64_t>(std::uint64_t scalar) { return _mm_set1_epi64x(std::bit_cast<long long>(scalar)); }
    template<> __m128i brocast<__m128i, std::int8_t>(std::int8_t scalar) { return _mm_set1_epi8(std::bit_cast<char>(scalar)); }
    template<> __m128i brocast<__m128i, std::int16_t>(std::int16_t scalar) { return _mm_set1_epi16(std::bit_cast<short>(scalar)); }
    template<> __m128i brocast<__m128i, std::int32_t>(std::int32_t scalar) { return _mm_set1_epi32(std::bit_cast<int>(scalar)); }
    template<> __m128i brocast<__m128i, std::int64_t>(std::int64_t scalar) { return _mm_set1_epi64x(std::bit_cast<long long>(scalar)); }
    template<> __m256i brocast<__m256i, std::uint8_t>(std::uint8_t scalar) { return _mm256_set1_epi8(std::bit_cast<char>(scalar)); }
    template<> __m256i brocast<__m256i, std::uint16_t>(std::uint16_t scalar) { return _mm256_set1_epi16(std::bit_cast<short>(scalar)); }
    template<> __m256i brocast<__m256i, std::uint32_t>(std::uint32_t scalar) { return _mm256_set1_epi32(std::bit_cast<int>(scalar)); }
    template<> __m256i brocast<__m256i, std::uint64_t>(std::uint64_t scalar) { return _mm256_set1_epi64x(std::bit_cast<long long>(scalar)); }
    template<> __m256i brocast<__m256i, std::int8_t>(std::int8_t scalar) { return _mm256_set1_epi8(std::bit_cast<char>(scalar)); }
    template<> __m256i brocast<__m256i, std::int16_t>(std::int16_t scalar) { return _mm256_set1_epi16(std::bit_cast<short>(scalar)); }
    template<> __m256i brocast<__m256i, std::int32_t>(std::int32_t scalar) { return _mm256_set1_epi32(std::bit_cast<int>(scalar)); }
    template<> __m256i brocast<__m256i, std::int64_t>(std::int64_t scalar) { return _mm256_set1_epi64x(std::bit_cast<long long>(scalar)); }
    template<> __m128 brocast<__m128, float>(float scalar) { return _mm_set1_ps(scalar); }
    template<> __m256 brocast<__m256, float>(float scalar) { return _mm256_set1_ps(scalar); }
    template<> __m128d brocast<__m128d, double>(double scalar) { return _mm_set1_pd(scalar); }
    template<> __m256d brocast<__m256d, double>(double scalar) { return _mm256_set1_pd(scalar); }


#ifdef _FOYE_SIMD_HAS_FP16_
    template<> __m128i brocast<__m128i, fy::float16>(fy::float16 scalar) { return _mm_set1_epi16(std::bit_cast<short>(scalar)); }
    template<> __m256i brocast<__m256i, fy::float16>(fy::float16 scalar) { return _mm256_set1_epi16(std::bit_cast<short>(scalar)); }
#endif

#ifdef _FOYE_SIMD_HAS_BF16_
    template<> __m128i brocast<__m128i, fy::bfloat16>(fy::bfloat16 scalar) { return _mm_set1_epi16(std::bit_cast<short>(scalar)); }
    template<> __m256i brocast<__m256i, fy::bfloat16>(fy::bfloat16 scalar) { return _mm256_set1_epi16(std::bit_cast<short>(scalar)); }
#endif

    template<std::size_t index> std::uint8_t extract_x8(__m128i src) { return static_cast<std::uint8_t>(_mm_extract_epi8(src, index)); }
    template<std::size_t index> std::uint16_t extract_x16(__m128i src) { return static_cast<std::uint16_t>(_mm_extract_epi16(src, index)); }
    template<std::size_t index> std::uint32_t extract_x32(__m128i src) { return std::bit_cast<std::uint32_t>(_mm_extract_epi32(src, index)); }
    template<std::size_t index> std::uint64_t extract_x64(__m128i src) { return std::bit_cast<std::uint64_t>(_mm_extract_epi64(src, index)); }

    template<std::size_t index> std::uint8_t extract_x8(__m256i src) { return static_cast<std::uint8_t>(_mm256_extract_epi8(src, index)); }
    template<std::size_t index> std::uint16_t extract_x16(__m256i src) { return static_cast<std::uint16_t>(_mm256_extract_epi16(src, index)); }
    template<std::size_t index> std::uint32_t extract_x32(__m256i src) { return std::bit_cast<std::uint32_t>(_mm256_extract_epi32(src, index)); }
    template<std::size_t index> std::uint64_t extract_x64(__m256i src) { return std::bit_cast<std::uint64_t>(_mm256_extract_epi64(src, index)); }

    template<std::size_t index> std::uint32_t extract_x32(__m128 src) { return std::bit_cast<std::uint32_t>(_mm_extract_epi32(_mm_castps_si128(src), index)); }
    template<std::size_t index> std::uint64_t extract_x64(__m128d src) { return std::bit_cast<std::uint64_t>(_mm_extract_epi64(_mm_castpd_si128(src), index)); }

    template<std::size_t index> std::uint32_t extract_x32(__m256 src) { return std::bit_cast<std::uint32_t>(_mm256_extract_epi32(_mm256_castps_si256(src), index)); }
    template<std::size_t index> std::uint64_t extract_x64(__m256d src) { return std::bit_cast<std::uint64_t>(_mm256_extract_epi64(_mm256_castpd_si256(src), index)); }


    template<typename dst, typename src> dst basic_reinterpret(src) { __assume(false); }
    template<> __m128 basic_reinterpret(__m128 v) { return v; }
    template<> __m128d basic_reinterpret(__m128d v) { return v; }
    template<> __m128i basic_reinterpret(__m128i v) { return v; }

    template<> __m128 basic_reinterpret(__m128i v) { return _mm_castsi128_ps(v); }
    template<> __m128d basic_reinterpret(__m128i v) { return _mm_castsi128_pd(v); }

    template<> __m128i basic_reinterpret(__m128 v) { return _mm_castps_si128(v); }
    template<> __m128d basic_reinterpret(__m128 v) { return _mm_castps_pd(v); }

    template<> __m128 basic_reinterpret(__m128d v) { return _mm_castpd_ps(v); }
    template<> __m128i basic_reinterpret(__m128d v) { return _mm_castpd_si128(v); }

    template<> __m256 basic_reinterpret(__m256 v) { return v; }
    template<> __m256d basic_reinterpret(__m256d v) { return v; }
    template<> __m256i basic_reinterpret(__m256i v) { return v; }

    template<> __m256 basic_reinterpret(__m256i v) { return _mm256_castsi256_ps(v); }
    template<> __m256d basic_reinterpret(__m256i v) { return _mm256_castsi256_pd(v); }

    template<> __m256i basic_reinterpret(__m256 v) { return _mm256_castps_si256(v); }
    template<> __m256d basic_reinterpret(__m256 v) { return _mm256_castps_pd(v); }

    template<> __m256 basic_reinterpret(__m256d v) { return _mm256_castpd_ps(v); }
    template<> __m256i basic_reinterpret(__m256d v) { return _mm256_castpd_si256(v); }
}




#endif