#ifndef _FOYE_SIMD_OPT_HPP_
#define _FOYE_SIMD_OPT_HPP_
#pragma once

#include "simd_def.hpp"
#include "simd_utility.hpp"
#include "simd_cvt.hpp"
#include "simd_shift.hpp"

namespace fyx::simd
{
#define DEFINE_BITWISE_OPERATION(NAME, intrin_suffix) \
template<typename T, std::size_t bits_width> \
basic_simd<T, bits_width> bitwise_##NAME(basic_simd<T, bits_width> lhs, basic_simd<T, bits_width> rhs) \
{ \
    using simd_type = basic_simd<T, bits_width>; \
    \
    if constexpr (std::is_same_v<typename simd_type::vector_t, __m128i> || \
                 std::is_same_v<typename basic_simd<T, 256>::vector_t, __m256i>) \
    { \
        if constexpr (bits_width == 128) \
            { return simd_type{ _mm_##intrin_suffix##_si128(lhs.data, rhs.data) }; } \
        else if constexpr (bits_width == 256) \
            { return simd_type{ _mm256_##intrin_suffix##_si256(lhs.data, rhs.data) }; } \
    } \
    else \
    { \
        if constexpr (std::is_same_v<float32x8, simd_type>) \
            { return float32x8(_mm256_##intrin_suffix##_ps(lhs.data, rhs.data)); } \
        else if constexpr (std::is_same_v<float64x4, simd_type>) \
            { return float64x4(_mm256_##intrin_suffix##_pd(lhs.data, rhs.data)); } \
        else if constexpr (std::is_same_v<float32x4, simd_type>) \
            { return float32x4(_mm_##intrin_suffix##_ps(lhs.data, rhs.data)); } \
        else if constexpr (std::is_same_v<float64x2, simd_type>) \
            { return float64x2(_mm_##intrin_suffix##_pd(lhs.data, rhs.data)); } \
    } \
}
    DEFINE_BITWISE_OPERATION(AND, and)
    DEFINE_BITWISE_OPERATION(OR, or )
    DEFINE_BITWISE_OPERATION(XOR, xor)
    DEFINE_BITWISE_OPERATION(ANDNOT, andnot)
#undef DEFINE_BITWISE_OPERATION

    template<typename T, std::size_t bits_width>
    basic_simd<T, bits_width> bitwise_NOT(basic_simd<T, bits_width> arg)
    {
        basic_simd<T, bits_width> allone{
            fyx::simd::detail::brocast<typename basic_simd<T, bits_width>::vector_t,
            typename basic_simd<T, bits_width>::scalar_t>(static_cast<T>(-1))
        };

        return basic_simd<T, bits_width>{ bitwise_XOR(arg, allone) };
    }
}

namespace fyx::simd
{
    uint8x16 plus(uint8x16 lhs, uint8x16 rhs) { return uint8x16{ _mm_add_epi8(lhs.data, rhs.data) }; }
    uint16x8 plus(uint16x8 lhs, uint16x8 rhs) { return uint16x8{ _mm_add_epi16(lhs.data, rhs.data) }; }
    uint32x4 plus(uint32x4 lhs, uint32x4 rhs) { return uint32x4{ _mm_add_epi32(lhs.data, rhs.data) }; }
    uint64x2 plus(uint64x2 lhs, uint64x2 rhs) { return uint64x2{ _mm_add_epi64(lhs.data, rhs.data) }; }
    uint8x32 plus(uint8x32 lhs, uint8x32 rhs) { return uint8x32{ _mm256_add_epi8(lhs.data, rhs.data) }; }
    uint16x16 plus(uint16x16 lhs, uint16x16 rhs) { return uint16x16{ _mm256_add_epi16(lhs.data, rhs.data) }; }
    uint32x8 plus(uint32x8 lhs, uint32x8 rhs) { return uint32x8{ _mm256_add_epi32(lhs.data, rhs.data) }; }
    uint64x4 plus(uint64x4 lhs, uint64x4 rhs) { return uint64x4{ _mm256_add_epi64(lhs.data, rhs.data) }; }
    sint8x16 plus(sint8x16 lhs, sint8x16 rhs) { return sint8x16{ _mm_add_epi8(lhs.data, rhs.data) }; }
    sint16x8 plus(sint16x8 lhs, sint16x8 rhs) { return sint16x8{ _mm_add_epi16(lhs.data, rhs.data) }; }
    sint32x4 plus(sint32x4 lhs, sint32x4 rhs) { return sint32x4{ _mm_add_epi32(lhs.data, rhs.data) }; }
    sint64x2 plus(sint64x2 lhs, sint64x2 rhs) { return sint64x2{ _mm_add_epi64(lhs.data, rhs.data) }; }
    sint8x32 plus(sint8x32 lhs, sint8x32 rhs) { return sint8x32{ _mm256_add_epi8(lhs.data, rhs.data) }; }
    sint16x16 plus(sint16x16 lhs, sint16x16 rhs) { return sint16x16{ _mm256_add_epi16(lhs.data, rhs.data) }; }
    sint32x8 plus(sint32x8 lhs, sint32x8 rhs) { return sint32x8{ _mm256_add_epi32(lhs.data, rhs.data) }; }
    sint64x4 plus(sint64x4 lhs, sint64x4 rhs) { return sint64x4{ _mm256_add_epi64(lhs.data, rhs.data) }; }
    float32x8 plus(float32x8 lhs, float32x8 rhs) { return float32x8{ _mm256_add_ps(lhs.data, rhs.data) }; }
    float64x4 plus(float64x4 lhs, float64x4 rhs) { return float64x4{ _mm256_add_pd(lhs.data, rhs.data) }; }
    float32x4 plus(float32x4 lhs, float32x4 rhs) { return float32x4{ _mm_add_ps(lhs.data, rhs.data) }; }
    float64x2 plus(float64x2 lhs, float64x2 rhs) { return float64x2{ _mm_add_pd(lhs.data, rhs.data) }; }
    _FOYE_SIMD_DISPATCH_8LANE_V_VV_(plus, _mm256_add_ps)

    uint8x16 minus(uint8x16 lhs, uint8x16 rhs) { return uint8x16{ _mm_sub_epi8(lhs.data, rhs.data) }; }
    uint16x8 minus(uint16x8 lhs, uint16x8 rhs) { return uint16x8{ _mm_sub_epi16(lhs.data, rhs.data) }; }
    uint32x4 minus(uint32x4 lhs, uint32x4 rhs) { return uint32x4{ _mm_sub_epi32(lhs.data, rhs.data) }; }
    uint64x2 minus(uint64x2 lhs, uint64x2 rhs) { return uint64x2{ _mm_sub_epi64(lhs.data, rhs.data) }; }
    uint8x32 minus(uint8x32 lhs, uint8x32 rhs) { return uint8x32{ _mm256_sub_epi8(lhs.data, rhs.data) }; }
    uint16x16 minus(uint16x16 lhs, uint16x16 rhs) { return uint16x16{ _mm256_sub_epi16(lhs.data, rhs.data) }; }
    uint32x8 minus(uint32x8 lhs, uint32x8 rhs) { return uint32x8{ _mm256_sub_epi32(lhs.data, rhs.data) }; }
    uint64x4 minus(uint64x4 lhs, uint64x4 rhs) { return uint64x4{ _mm256_sub_epi64(lhs.data, rhs.data) }; }
    sint8x16 minus(sint8x16 lhs, sint8x16 rhs) { return sint8x16{ _mm_sub_epi8(lhs.data, rhs.data) }; }
    sint16x8 minus(sint16x8 lhs, sint16x8 rhs) { return sint16x8{ _mm_sub_epi16(lhs.data, rhs.data) }; }
    sint32x4 minus(sint32x4 lhs, sint32x4 rhs) { return sint32x4{ _mm_sub_epi32(lhs.data, rhs.data) }; }
    sint64x2 minus(sint64x2 lhs, sint64x2 rhs) { return sint64x2{ _mm_sub_epi64(lhs.data, rhs.data) }; }
    sint8x32 minus(sint8x32 lhs, sint8x32 rhs) { return sint8x32{ _mm256_sub_epi8(lhs.data, rhs.data) }; }
    sint16x16 minus(sint16x16 lhs, sint16x16 rhs) { return sint16x16{ _mm256_sub_epi16(lhs.data, rhs.data) }; }
    sint32x8 minus(sint32x8 lhs, sint32x8 rhs) { return sint32x8{ _mm256_sub_epi32(lhs.data, rhs.data) }; }
    sint64x4 minus(sint64x4 lhs, sint64x4 rhs) { return sint64x4{ _mm256_sub_epi64(lhs.data, rhs.data) }; }
    float32x8 minus(float32x8 lhs, float32x8 rhs) { return float32x8{ _mm256_sub_ps(lhs.data, rhs.data) }; }
    float64x4 minus(float64x4 lhs, float64x4 rhs) { return float64x4{ _mm256_sub_pd(lhs.data, rhs.data) }; }
    float32x4 minus(float32x4 lhs, float32x4 rhs) { return float32x4{ _mm_sub_ps(lhs.data, rhs.data) }; }
    float64x2 minus(float64x2 lhs, float64x2 rhs) { return float64x2{ _mm_sub_pd(lhs.data, rhs.data) }; }
    _FOYE_SIMD_DISPATCH_8LANE_V_VV_(minus, _mm256_sub_ps)

    uint8x16 divide(uint8x16 lhs, uint8x16 rhs) { return uint8x16{ _mm_div_epu8(lhs.data, rhs.data) }; }
    uint16x8 divide(uint16x8 lhs, uint16x8 rhs) { return uint16x8{ _mm_div_epu16(lhs.data, rhs.data) }; }
    uint32x4 divide(uint32x4 lhs, uint32x4 rhs) { return uint32x4{ _mm_div_epu32(lhs.data, rhs.data) }; }
    uint64x2 divide(uint64x2 lhs, uint64x2 rhs) { return uint64x2{ _mm_div_epu64(lhs.data, rhs.data) }; }
    uint8x32 divide(uint8x32 lhs, uint8x32 rhs) { return uint8x32{ _mm256_div_epu8(lhs.data, rhs.data) }; }
    uint16x16 divide(uint16x16 lhs, uint16x16 rhs) { return uint16x16{ _mm256_div_epu16(lhs.data, rhs.data) }; }
    uint32x8 divide(uint32x8 lhs, uint32x8 rhs) { return uint32x8{ _mm256_div_epu32(lhs.data, rhs.data) }; }
    uint64x4 divide(uint64x4 lhs, uint64x4 rhs) { return uint64x4{ _mm256_div_epu64(lhs.data, rhs.data) }; }
    sint8x16 divide(sint8x16 lhs, sint8x16 rhs) { return sint8x16{ _mm_div_epi8(lhs.data, rhs.data) }; }
    sint16x8 divide(sint16x8 lhs, sint16x8 rhs) { return sint16x8{ _mm_div_epi16(lhs.data, rhs.data) }; }
    sint32x4 divide(sint32x4 lhs, sint32x4 rhs) { return sint32x4{ _mm_div_epi32(lhs.data, rhs.data) }; }
    sint64x2 divide(sint64x2 lhs, sint64x2 rhs) { return sint64x2{ _mm_div_epi64(lhs.data, rhs.data) }; }
    sint8x32 divide(sint8x32 lhs, sint8x32 rhs) { return sint8x32{ _mm256_div_epi8(lhs.data, rhs.data) }; }
    sint16x16 divide(sint16x16 lhs, sint16x16 rhs) { return sint16x16{ _mm256_div_epi16(lhs.data, rhs.data) }; }
    sint32x8 divide(sint32x8 lhs, sint32x8 rhs) { return sint32x8{ _mm256_div_epi32(lhs.data, rhs.data) }; }
    sint64x4 divide(sint64x4 lhs, sint64x4 rhs) { return sint64x4{ _mm256_div_epi64(lhs.data, rhs.data) }; }
    float32x8 divide(float32x8 lhs, float32x8 rhs) { return float32x8{ _mm256_div_ps(lhs.data, rhs.data) }; }
    float64x4 divide(float64x4 lhs, float64x4 rhs) { return float64x4{ _mm256_div_pd(lhs.data, rhs.data) }; }
    float32x4 divide(float32x4 lhs, float32x4 rhs) { return float32x4{ _mm_div_ps(lhs.data, rhs.data) }; }
    float64x2 divide(float64x2 lhs, float64x2 rhs) { return float64x2{ _mm_div_pd(lhs.data, rhs.data) }; }
    _FOYE_SIMD_DISPATCH_8LANE_V_VV_(divide, _mm256_div_ps)

    uint16x8 multiplies(uint16x8 lhs, uint16x8 rhs) { return uint16x8{ _mm_mullo_epi16(lhs.data, rhs.data) }; }
    uint32x4 multiplies(uint32x4 lhs, uint32x4 rhs) { return uint32x4{ _mm_mullo_epi32(lhs.data, rhs.data) }; }
    uint16x16 multiplies(uint16x16 lhs, uint16x16 rhs) { return uint16x16{ _mm256_mullo_epi16(lhs.data, rhs.data) }; }
    uint32x8 multiplies(uint32x8 lhs, uint32x8 rhs) { return uint32x8{ _mm256_mullo_epi32(lhs.data, rhs.data) }; }
    sint16x8 multiplies(sint16x8 lhs, sint16x8 rhs) { return sint16x8{ _mm_mullo_epi16(lhs.data, rhs.data) }; }
    sint32x4 multiplies(sint32x4 lhs, sint32x4 rhs) { return sint32x4{ _mm_mullo_epi32(lhs.data, rhs.data) }; }
    sint16x16 multiplies(sint16x16 lhs, sint16x16 rhs) { return sint16x16{ _mm256_mullo_epi16(lhs.data, rhs.data) }; }
    sint32x8 multiplies(sint32x8 lhs, sint32x8 rhs) { return sint32x8{ _mm256_mullo_epi32(lhs.data, rhs.data) }; }
    float32x8 multiplies(float32x8 lhs, float32x8 rhs) { return float32x8{ _mm256_mul_ps(lhs.data, rhs.data) }; }
    float64x4 multiplies(float64x4 lhs, float64x4 rhs) { return float64x4{ _mm256_mul_pd(lhs.data, rhs.data) }; }
    float32x4 multiplies(float32x4 lhs, float32x4 rhs) { return float32x4{ _mm_mul_ps(lhs.data, rhs.data) }; }
    float64x2 multiplies(float64x2 lhs, float64x2 rhs) { return float64x2{ _mm_mul_pd(lhs.data, rhs.data) }; }
    _FOYE_SIMD_DISPATCH_8LANE_V_VV_(multiplies, _mm256_mul_ps)

#if defined(_FOYE_SIMD_ENABLE_EMULATED_)
    sint8x16 multiplies(sint8x16 lhs, sint8x16 rhs)
    {
        return narrowing<sint8x16>(
            multiplies(expand<sint16x16>(lhs), expand<sint16x16>(rhs)));
    }

    sint8x32 multiplies(sint8x32 lhs, sint8x32 rhs)
    {
        sint8x16 result_low = narrowing<sint8x16>(
            multiplies(expand_low<sint16x16>(lhs), expand_low<sint16x16>(rhs)));

        sint8x16 result_high = narrowing<sint8x16>(
            multiplies(expand_high<sint16x16>(lhs), expand_high<sint16x16>(rhs)));
        return merge(result_low, result_high);
    }

    uint8x16 multiplies(uint8x16 lhs, uint8x16 rhs)
    {
        sint16x16 sv16l = expand<sint16x16>(lhs);
        sint16x16 sv16r = expand<sint16x16>(rhs);
        sint16x16 sv16res = multiplies(sv16l, sv16r);
        return narrowing<uint8x16>(bitwise_AND(sv16res, load_brocast<sint16x16>(255)));
    }

    uint8x32 multiplies(uint8x32 lhs, uint8x32 rhs)
    {
        uint8x16 result_low = narrowing<uint8x16>(
            multiplies(expand_low<sint16x16>(lhs), expand_low<sint16x16>(rhs)));

        uint8x16 result_high = narrowing<uint8x16>(
            multiplies(expand_high<sint16x16>(lhs), expand_high<sint16x16>(rhs)));
        return merge(result_low, result_high);
    }

    namespace detail
    {
        template<typename simd_type>
        simd_type multiplies_fallback64bits(simd_type lhs, simd_type rhs)
        {
            alignas(alignof(typename simd_type::vector_t))
                typename simd_type::scalar_t temp[simd_type::lane_width * 2] = {};
            fyx::simd::store_aligned(lhs, temp + 0);
            fyx::simd::store_aligned(rhs, temp + simd_type::lane_width);
            std::multiplies<typename simd_type::scalar_t> invoker{};
            for (std::size_t i = 0; i < simd_type::lane_width; ++i)
            {
                temp[i] = invoker(temp[i], temp[i + simd_type::lane_width]);
            }
            return fyx::simd::load_aligned<simd_type>(temp);
        }
    }

    sint64x4 multiplies(sint64x4 lhs, sint64x4 rhs) { return fyx::simd::detail::multiplies_fallback64bits<sint64x4>(lhs, rhs); }
    uint64x4 multiplies(uint64x4 lhs, uint64x4 rhs) { return fyx::simd::detail::multiplies_fallback64bits<uint64x4>(lhs, rhs); }
    sint64x2 multiplies(sint64x2 lhs, sint64x2 rhs) { return fyx::simd::detail::multiplies_fallback64bits<sint64x2>(lhs, rhs); }
    uint64x2 multiplies(uint64x2 lhs, uint64x2 rhs) { return fyx::simd::detail::multiplies_fallback64bits<uint64x2>(lhs, rhs); }
#endif
    
    uint8x16 remainder(uint8x16 lhs, uint8x16 rhs) { return uint8x16{ _mm_rem_epu8(lhs.data, rhs.data) }; }
    uint16x8 remainder(uint16x8 lhs, uint16x8 rhs) { return uint16x8{ _mm_rem_epu16(lhs.data, rhs.data) }; }
    uint32x4 remainder(uint32x4 lhs, uint32x4 rhs) { return uint32x4{ _mm_rem_epu32(lhs.data, rhs.data) }; }
    uint64x2 remainder(uint64x2 lhs, uint64x2 rhs) { return uint64x2{ _mm_rem_epu64(lhs.data, rhs.data) }; }
    uint8x32 remainder(uint8x32 lhs, uint8x32 rhs) { return uint8x32{ _mm256_rem_epu8(lhs.data, rhs.data) }; }
    uint16x16 remainder(uint16x16 lhs, uint16x16 rhs) { return uint16x16{ _mm256_rem_epu16(lhs.data, rhs.data) }; }
    uint32x8 remainder(uint32x8 lhs, uint32x8 rhs) { return uint32x8{ _mm256_rem_epu32(lhs.data, rhs.data) }; }
    uint64x4 remainder(uint64x4 lhs, uint64x4 rhs) { return uint64x4{ _mm256_rem_epu64(lhs.data, rhs.data) }; }
    sint8x16 remainder(sint8x16 lhs, sint8x16 rhs) { return sint8x16{ _mm_rem_epi8(lhs.data, rhs.data) }; }
    sint16x8 remainder(sint16x8 lhs, sint16x8 rhs) { return sint16x8{ _mm_rem_epi16(lhs.data, rhs.data) }; }
    sint32x4 remainder(sint32x4 lhs, sint32x4 rhs) { return sint32x4{ _mm_rem_epi32(lhs.data, rhs.data) }; }
    sint64x2 remainder(sint64x2 lhs, sint64x2 rhs) { return sint64x2{ _mm_rem_epi64(lhs.data, rhs.data) }; }
    sint8x32 remainder(sint8x32 lhs, sint8x32 rhs) { return sint8x32{ _mm256_rem_epi8(lhs.data, rhs.data) }; }
    sint16x16 remainder(sint16x16 lhs, sint16x16 rhs) { return sint16x16{ _mm256_rem_epi16(lhs.data, rhs.data) }; }
    sint32x8 remainder(sint32x8 lhs, sint32x8 rhs) { return sint32x8{ _mm256_rem_epi32(lhs.data, rhs.data) }; }
    sint64x4 remainder(sint64x4 lhs, sint64x4 rhs) { return sint64x4{ _mm256_rem_epi64(lhs.data, rhs.data) }; }

    sint8x16 negate(sint8x16 input) { return minus(allzero_bits_as<sint8x16>(), input); }
    sint16x8 negate(sint16x8 input) { return minus(allzero_bits_as<sint16x8>(), input); }
    sint32x4 negate(sint32x4 input) { return minus(allzero_bits_as<sint32x4>(), input); }
    sint64x2 negate(sint64x2 input) { return minus(allzero_bits_as<sint64x2>(), input); }

    sint8x32 negate(sint8x32 input) { return minus(allzero_bits_as<sint8x32>(), input); }
    sint16x16 negate(sint16x16 input) { return minus(allzero_bits_as<sint16x16>(), input); }
    sint32x8 negate(sint32x8 input) { return minus(allzero_bits_as<sint32x8>(), input); }
    sint64x4 negate(sint64x4 input) { return minus(allzero_bits_as<sint64x4>(), input); }

    float32x8 negate(float32x8 input) { return bitwise_XOR(input, load_brocast<float32x8>(-0.0f)); }
    float32x4 negate(float32x4 input) { return bitwise_XOR(input, load_brocast<float32x4>(-0.0f)); }
    float64x4 negate(float64x4 input) { return bitwise_XOR(input, load_brocast<float64x4>(-0.0)); }
    float64x2 negate(float64x2 input) { return bitwise_XOR(input, load_brocast<float64x2>(-0.0)); }

#if defined(_FOYE_SIMD_HAS_FP16_)
    float16x8 negate(float16x8 input) { return bitwise_XOR(input, reinterpret<float16x8>(load_brocast<uint16x8>(0b1000000000000000))); }
    float16x16 negate(float16x16 input) { return bitwise_XOR(input, reinterpret<float16x16>(load_brocast<uint16x16>(0b1000000000000000))); }
#endif
#if defined(_FOYE_SIMD_HAS_FP16_)
    bfloat16x8 negate(bfloat16x8 input) { return bitwise_XOR(input, reinterpret<bfloat16x8>(load_brocast<uint16x8>(0b1000000000000000))); }
    bfloat16x16 negate(bfloat16x16 input) { return bitwise_XOR(input, reinterpret<bfloat16x16>(load_brocast<uint16x16>(0b1000000000000000))); }
#endif
}

namespace fyx::simd
{
    uint8x16 min(uint8x16 lhs, uint8x16 rhs) { return uint8x16(_mm_min_epu8(lhs.data, rhs.data)); }
    uint16x8 min(uint16x8 lhs, uint16x8 rhs) { return uint16x8(_mm_min_epu16(lhs.data, rhs.data)); }
    uint32x4 min(uint32x4 lhs, uint32x4 rhs) { return uint32x4(_mm_min_epu32(lhs.data, rhs.data)); }
    uint8x32 min(uint8x32 lhs, uint8x32 rhs) { return uint8x32(_mm256_min_epu8(lhs.data, rhs.data)); }
    uint16x16 min(uint16x16 lhs, uint16x16 rhs) { return uint16x16(_mm256_min_epu16(lhs.data, rhs.data)); }
    uint32x8 min(uint32x8 lhs, uint32x8 rhs) { return uint32x8(_mm256_min_epu32(lhs.data, rhs.data)); }
    sint8x16 min(sint8x16 lhs, sint8x16 rhs) { return sint8x16(_mm_min_epi8(lhs.data, rhs.data)); }
    sint16x8 min(sint16x8 lhs, sint16x8 rhs) { return sint16x8(_mm_min_epi16(lhs.data, rhs.data)); }
    sint32x4 min(sint32x4 lhs, sint32x4 rhs) { return sint32x4(_mm_min_epi32(lhs.data, rhs.data)); }
    sint8x32 min(sint8x32 lhs, sint8x32 rhs) { return sint8x32(_mm256_min_epi8(lhs.data, rhs.data)); }
    sint16x16 min(sint16x16 lhs, sint16x16 rhs) { return sint16x16(_mm256_min_epi16(lhs.data, rhs.data)); }
    sint32x8 min(sint32x8 lhs, sint32x8 rhs) { return sint32x8(_mm256_min_epi32(lhs.data, rhs.data)); }
    float32x4 min(float32x4 lhs, float32x4 rhs) { return float32x4(_mm_min_ps(lhs.data, rhs.data)); }
    float32x8 min(float32x8 lhs, float32x8 rhs) { return float32x8(_mm256_min_ps(lhs.data, rhs.data)); }
    float64x2 min(float64x2 lhs, float64x2 rhs) { return float64x2(_mm_min_pd(lhs.data, rhs.data)); }
    float64x4 min(float64x4 lhs, float64x4 rhs) { return float64x4(_mm256_min_pd(lhs.data, rhs.data)); }
    _FOYE_SIMD_DISPATCH_8LANE_V_VV_(min, _mm256_min_ps)

    uint8x16 max(uint8x16 lhs, uint8x16 rhs) { return uint8x16(_mm_max_epu8(lhs.data, rhs.data)); }
    uint16x8 max(uint16x8 lhs, uint16x8 rhs) { return uint16x8(_mm_max_epu16(lhs.data, rhs.data)); }
    uint32x4 max(uint32x4 lhs, uint32x4 rhs) { return uint32x4(_mm_max_epu32(lhs.data, rhs.data)); }
    uint8x32 max(uint8x32 lhs, uint8x32 rhs) { return uint8x32(_mm256_max_epu8(lhs.data, rhs.data)); }
    uint16x16 max(uint16x16 lhs, uint16x16 rhs) { return uint16x16(_mm256_max_epu16(lhs.data, rhs.data)); }
    uint32x8 max(uint32x8 lhs, uint32x8 rhs) { return uint32x8(_mm256_max_epu32(lhs.data, rhs.data)); }
    sint8x16 max(sint8x16 lhs, sint8x16 rhs) { return sint8x16(_mm_max_epi8(lhs.data, rhs.data)); }
    sint16x8 max(sint16x8 lhs, sint16x8 rhs) { return sint16x8(_mm_max_epi16(lhs.data, rhs.data)); }
    sint32x4 max(sint32x4 lhs, sint32x4 rhs) { return sint32x4(_mm_max_epi32(lhs.data, rhs.data)); }
    sint8x32 max(sint8x32 lhs, sint8x32 rhs) { return sint8x32(_mm256_max_epi8(lhs.data, rhs.data)); }
    sint16x16 max(sint16x16 lhs, sint16x16 rhs) { return sint16x16(_mm256_max_epi16(lhs.data, rhs.data)); }
    sint32x8 max(sint32x8 lhs, sint32x8 rhs) { return sint32x8(_mm256_max_epi32(lhs.data, rhs.data)); }
    float32x4 max(float32x4 lhs, float32x4 rhs) { return float32x4(_mm_max_ps(lhs.data, rhs.data)); }
    float32x8 max(float32x8 lhs, float32x8 rhs) { return float32x8(_mm256_max_ps(lhs.data, rhs.data)); }
    float64x2 max(float64x2 lhs, float64x2 rhs) { return float64x2(_mm_max_pd(lhs.data, rhs.data)); }
    float64x4 max(float64x4 lhs, float64x4 rhs) { return float64x4(_mm256_max_pd(lhs.data, rhs.data)); }
    _FOYE_SIMD_DISPATCH_8LANE_V_VV_(max, _mm256_max_ps)


#if defined(_FOYE_SIMD_ENABLE_EMULATED_)
#define DEFINE_MINMAX_FALLBACK(input_simd_type, cmpfunc) \
input_simd_type cmpfunc(input_simd_type lhs, input_simd_type rhs)\
{\
    alignas(alignof(typename input_simd_type::vector_t))\
        typename input_simd_type::scalar_t temp[input_simd_type::lane_width * 2];\
    fyx::simd::store_aligned(lhs, temp + 0);\
    fyx::simd::store_aligned(rhs, temp + input_simd_type::lane_width);\
    for (std::size_t i = 0; i < input_simd_type::lane_width; ++i)\
    {\
        temp[i] = (::std::cmpfunc)(temp[i], temp[i + input_simd_type::lane_width]);\
    }\
    return fyx::simd::load_aligned<input_simd_type>(temp);\
}
#define DEFINE_MINMAX_FALLBACK_2WAY_DISPATCH(input_simd_type) \
    DEFINE_MINMAX_FALLBACK(input_simd_type, min) \
    DEFINE_MINMAX_FALLBACK(input_simd_type, max)

    DEFINE_MINMAX_FALLBACK_2WAY_DISPATCH(uint64x2)
    DEFINE_MINMAX_FALLBACK_2WAY_DISPATCH(sint64x2)
    DEFINE_MINMAX_FALLBACK_2WAY_DISPATCH(uint64x4)
    DEFINE_MINMAX_FALLBACK_2WAY_DISPATCH(sint64x4)
#undef DEFINE_MINMAX_FALLBACK_2WAY_DISPATCH
#undef DEFINE_MINMAX_FALLBACK
#endif

    uint8x16 clamp(uint8x16 input, uint8x16 minval, uint8x16 maxval) { return uint8x16{ fyx::simd::min(fyx::simd::max(input, minval), maxval) }; }
    uint16x8 clamp(uint16x8 input, uint16x8 minval, uint16x8 maxval) { return uint16x8{ fyx::simd::min(fyx::simd::max(input, minval), maxval) }; }
    uint32x4 clamp(uint32x4 input, uint32x4 minval, uint32x4 maxval) { return uint32x4{ fyx::simd::min(fyx::simd::max(input, minval), maxval) }; }
    uint64x2 clamp(uint64x2 input, uint64x2 minval, uint64x2 maxval) { return uint64x2{ fyx::simd::min(fyx::simd::max(input, minval), maxval) }; }
    sint8x16 clamp(sint8x16 input, sint8x16 minval, sint8x16 maxval) { return sint8x16{ fyx::simd::min(fyx::simd::max(input, minval), maxval) }; }
    sint16x8 clamp(sint16x8 input, sint16x8 minval, sint16x8 maxval) { return sint16x8{ fyx::simd::min(fyx::simd::max(input, minval), maxval) }; }
    sint32x4 clamp(sint32x4 input, sint32x4 minval, sint32x4 maxval) { return sint32x4{ fyx::simd::min(fyx::simd::max(input, minval), maxval) }; }
    sint64x2 clamp(sint64x2 input, sint64x2 minval, sint64x2 maxval) { return sint64x2{ fyx::simd::min(fyx::simd::max(input, minval), maxval) }; }
    float32x4 clamp(float32x4 input, float32x4 minval, float32x4 maxval) { return float32x4{ fyx::simd::min(fyx::simd::max(input, minval), maxval) }; }
    float64x2 clamp(float64x2 input, float64x2 minval, float64x2 maxval) { return float64x2{ fyx::simd::min(fyx::simd::max(input, minval), maxval) }; }

    uint8x32 clamp(uint8x32 input, uint8x32 minval, uint8x32 maxval) { return uint8x32{ fyx::simd::min(fyx::simd::max(input, minval), maxval) }; }
    uint16x16 clamp(uint16x16 input, uint16x16 minval, uint16x16 maxval) { return uint16x16{ fyx::simd::min(fyx::simd::max(input, minval), maxval) }; }
    uint32x8 clamp(uint32x8 input, uint32x8 minval, uint32x8 maxval) { return uint32x8{ fyx::simd::min(fyx::simd::max(input, minval), maxval) }; }
    uint64x4 clamp(uint64x4 input, uint64x4 minval, uint64x4 maxval) { return uint64x4{ fyx::simd::min(fyx::simd::max(input, minval), maxval) }; }
    sint8x32 clamp(sint8x32 input, sint8x32 minval, sint8x32 maxval) { return sint8x32{ fyx::simd::min(fyx::simd::max(input, minval), maxval) }; }
    sint16x16 clamp(sint16x16 input, sint16x16 minval, sint16x16 maxval) { return sint16x16{ fyx::simd::min(fyx::simd::max(input, minval), maxval) }; }
    sint32x8 clamp(sint32x8 input, sint32x8 minval, sint32x8 maxval) { return sint32x8{ fyx::simd::min(fyx::simd::max(input, minval), maxval) }; }
    sint64x4 clamp(sint64x4 input, sint64x4 minval, sint64x4 maxval) { return sint64x4{ fyx::simd::min(fyx::simd::max(input, minval), maxval) }; }
    float32x8 clamp(float32x8 input, float32x8 minval, float32x8 maxval) { return float32x8{ fyx::simd::min(fyx::simd::max(input, minval), maxval) }; }
    float64x4 clamp(float64x4 input, float64x4 minval, float64x4 maxval) { return float64x4{ fyx::simd::min(fyx::simd::max(input, minval), maxval) }; }

#if defined(_FOYE_SIMD_HAS_FP16_)
    float16x8 clamp(float16x8 input, float16x8 minval, float16x8 maxval) 
    { return float16x8{ fyx::simd::min(fyx::simd::max(input, minval), maxval) }; }

    float16x16 clamp(float16x16 input, float16x16 minval, float16x16 maxval)
    { return float16x16{ fyx::simd::min(fyx::simd::max(input, minval), maxval) }; }
#endif
#if defined(_FOYE_SIMD_HAS_FP16_)
    bfloat16x8 clamp(bfloat16x8 input, bfloat16x8 minval, bfloat16x8 maxval)
    { return bfloat16x8{ fyx::simd::min(fyx::simd::max(input, minval), maxval) }; }

    bfloat16x16 clamp(bfloat16x16 input, bfloat16x16 minval, bfloat16x16 maxval)
    { return bfloat16x16{ fyx::simd::min(fyx::simd::max(input, minval), maxval) }; }
#endif

    FOYE_SIMD_ERROR_WHEN_CALLED("No one needs the absolute value of an unsigned integer") uint8x16 abs(uint8x16 input) { return input; }
    FOYE_SIMD_ERROR_WHEN_CALLED("No one needs the absolute value of an unsigned integer") uint16x8 abs(uint16x8 input) { return input; }
    FOYE_SIMD_ERROR_WHEN_CALLED("No one needs the absolute value of an unsigned integer") uint32x4 abs(uint32x4 input) { return input; }
    FOYE_SIMD_ERROR_WHEN_CALLED("No one needs the absolute value of an unsigned integer") uint64x2 abs(uint64x2 input) { return input; }
    FOYE_SIMD_ERROR_WHEN_CALLED("No one needs the absolute value of an unsigned integer") uint8x32 abs(uint8x32 input) { return input; }
    FOYE_SIMD_ERROR_WHEN_CALLED("No one needs the absolute value of an unsigned integer") uint16x16 abs(uint16x16 input) { return input; }
    FOYE_SIMD_ERROR_WHEN_CALLED("No one needs the absolute value of an unsigned integer") uint32x8 abs(uint32x8 input) { return input; }
    FOYE_SIMD_ERROR_WHEN_CALLED("No one needs the absolute value of an unsigned integer") uint64x4 abs(uint64x4 input) { return input; }

    sint8x16 abs(sint8x16 input) { return sint8x16{ _mm_abs_epi8(input.data) }; }
    sint16x8 abs(sint16x8 input) { return sint16x8{ _mm_abs_epi16(input.data) }; }
    sint32x4 abs(sint32x4 input) { return sint32x4{ _mm_abs_epi32(input.data) }; }
    sint8x32 abs(sint8x32 input) { return sint8x32{ _mm256_abs_epi8(input.data) }; }
    sint16x16 abs(sint16x16 input) { return sint16x16{ _mm256_abs_epi16(input.data) }; }
    sint32x8 abs(sint32x8 input) { return sint32x8{ _mm256_abs_epi32(input.data) }; }

#if defined(_FOYE_SIMD_ENABLE_EMULATED_)
    sint64x2 abs(sint64x2 input)
    {
        __m128i sign_mask = _mm_cmpgt_epi64(_mm_setzero_si128(), input.data);
        __m128i abs_val = _mm_add_epi64(
            _mm_xor_si128(input.data, sign_mask),
            _mm_srli_epi64(sign_mask, 63));
        return sint64x2{ abs_val };
    }

    sint64x4 abs(sint64x4 input)
    {
        __m256i sign_mask = _mm256_cmpgt_epi64(_mm256_setzero_si256(), input.data);
        __m256i abs_val = _mm256_add_epi64(
            _mm256_xor_si256(input.data, sign_mask),
            _mm256_srli_epi64(sign_mask, 63));
        return sint64x4{ abs_val };
    }

    float32x8 abs(float32x8 input) { return float32x8{ _mm256_andnot_ps(_mm256_set1_ps(-0.0f), input.data) }; }
    float32x4 abs(float32x4 input) { return float32x4{ _mm_andnot_ps(_mm_set1_ps(-0.0f), input.data) }; }
    float64x4 abs(float64x4 input) { return float64x4{ _mm256_andnot_pd(_mm256_set1_pd(-0.0), input.data) }; }
    float64x2 abs(float64x2 input) { return float64x2{ _mm_andnot_pd(_mm_set1_pd(-0.0), input.data) }; }

    _FOYE_SIMD_DISPATCH_8LANE_V_V_(abs, _FOYE_SIMD_ABS_PS_)
#endif

    uint8x16 avg(uint8x16 arg0, uint8x16 arg1) { return uint8x16{ _mm_avg_epu8(arg0.data, arg1.data) }; }
    uint16x8 avg(uint16x8 arg0, uint8x16 arg1) { return uint16x8{ _mm_avg_epu16(arg0.data, arg1.data) }; }
    uint8x32 avg(uint8x32 arg0, uint8x32 arg1) { return uint8x32{ _mm256_avg_epu8(arg0.data, arg1.data) }; }
    uint16x16 avg(uint16x16 arg0, uint16x16 arg1) { return uint16x16{ _mm256_avg_epu16(arg0.data, arg1.data) }; }
#if defined(_FOYE_SIMD_ENABLE_EMULATED_)
    float32x8 avg(float32x8 arg0, float32x8 arg1)
    {
        return float32x8{ _mm256_fmadd_ps(_mm256_add_ps(arg0.data, arg1.data),
            _mm256_set1_ps(0.5f), _mm256_setzero_ps()) };
    }

    float32x4 avg(float32x4 arg0, float32x4 arg1)
    {
        return float32x4{ _mm_fmadd_ps(_mm_add_ps(arg0.data, arg1.data),
            _mm_set1_ps(0.5f), _mm_setzero_ps()) };
    }

    float64x4 avg(float64x4 arg0, float64x4 arg1)
    {
        return float64x4{ _mm256_fmadd_pd(_mm256_add_pd(arg0.data, arg1.data),
            _mm256_set1_pd(0.5), _mm256_setzero_pd()) };
    }

    float64x2 avg(float64x2 arg0, float64x2 arg1)
    {
        return float64x2{ _mm_fmadd_pd(_mm_add_pd(arg0.data, arg1.data),
            _mm_set1_pd(0.5), _mm_setzero_pd()) };
    }

    namespace detail
    {
        template<typename simd_type>
        simd_type avg_fallback(simd_type arg0, simd_type arg1)
        {
            simd_type xorres = fyx::simd::bitwise_XOR(arg0, arg1);
            xorres = fyx::simd::shift_right<1>(xorres);
            simd_type res = fyx::simd::plus(fyx::simd::bitwise_AND(arg0, arg1), xorres);
            return simd_type{ res };
        }
    }

    uint32x4 avg(uint32x4 arg0, uint32x4 arg1) { return fyx::simd::detail::avg_fallback<uint32x4>(arg0, arg1); }
    uint64x2 avg(uint64x2 arg0, uint64x2 arg1) { return fyx::simd::detail::avg_fallback<uint64x2>(arg0, arg1); }
    uint32x8 avg(uint32x8 arg0, uint32x8 arg1) { return fyx::simd::detail::avg_fallback<uint32x8>(arg0, arg1); }
    uint64x4 avg(uint64x4 arg0, uint64x4 arg1) { return fyx::simd::detail::avg_fallback<uint64x4>(arg0, arg1); }
    sint8x16 avg(sint8x16 arg0, sint8x16 arg1) { return fyx::simd::detail::avg_fallback<sint8x16>(arg0, arg1); }
    sint16x8 avg(sint16x8 arg0, sint16x8 arg1) { return fyx::simd::detail::avg_fallback<sint16x8>(arg0, arg1); }
    sint32x4 avg(sint32x4 arg0, sint32x4 arg1) { return fyx::simd::detail::avg_fallback<sint32x4>(arg0, arg1); }
    sint64x2 avg(sint64x2 arg0, sint64x2 arg1) { return fyx::simd::detail::avg_fallback<sint64x2>(arg0, arg1); }
    sint8x32 avg(sint8x32 arg0, sint8x32 arg1) { return fyx::simd::detail::avg_fallback<sint8x32>(arg0, arg1); }
    sint16x16 avg(sint16x16 arg0, sint16x16 arg1) { return fyx::simd::detail::avg_fallback<sint16x16>(arg0, arg1); }
    sint32x8 avg(sint32x8 arg0, sint32x8 arg1) { return fyx::simd::detail::avg_fallback<sint32x8>(arg0, arg1); }
    sint64x4 avg(sint64x4 arg0, sint64x4 arg1) { return fyx::simd::detail::avg_fallback<sint64x4>(arg0, arg1); }

#if defined(_FOYE_SIMD_HAS_FP16_)
    float16x8 avg(float16x8 arg0, float16x8 arg1)
    {
        __m256 lhs = cvt8lane_fp16_to_fp32(arg0.data);
        __m256 rhs = cvt8lane_fp16_to_fp32(arg1.data);
        __m256 res = _mm256_fmadd_ps(_mm256_add_ps(lhs, rhs),
            _mm256_set1_ps(0.5f), _mm256_setzero_ps());
        return float16x8{ cvt8lane_fp32_to_fp16(res) };
    }

    float16x16 avg(float16x16 arg0, float16x16 arg1)
    {
        __m256i vlhs = arg0.data;
        __m256i vrhs = arg1.data;

        __m256 lhs_low = cvt8lane_fp16_to_fp32(detail::split_low(vlhs));
        __m256 lhs_high = cvt8lane_fp16_to_fp32(detail::split_high(vlhs));
        __m256 rhs_low = cvt8lane_fp16_to_fp32(detail::split_low(vrhs));
        __m256 rhs_high = cvt8lane_fp16_to_fp32(detail::split_high(vrhs));
        const __m256 vzero = _mm256_setzero_ps();
        const __m256 vhalf = _mm256_set1_ps(0.5f);
        __m256 res_low = _mm256_fmadd_ps(_mm256_add_ps(lhs_low, rhs_low), vhalf, vzero);
        __m256 res_high = _mm256_fmadd_ps(_mm256_add_ps(lhs_high, rhs_high), vhalf, vzero);

        return float16x16{ detail::merge(
            cvt8lane_fp32_to_fp16(res_low),
            cvt8lane_fp32_to_fp16(res_high)) };
    }
#endif
#if defined(_FOYE_SIMD_HAS_FP16_)
    bfloat16x8 avg(bfloat16x8 arg0, bfloat16x8 arg1)
    {
        __m256 lhs = cvt8lane_bf16_to_fp32(arg0.data);
        __m256 rhs = cvt8lane_bf16_to_fp32(arg1.data);
        __m256 res = _mm256_fmadd_ps(_mm256_add_ps(lhs, rhs),
            _mm256_set1_ps(0.5f), _mm256_setzero_ps());
        return bfloat16x8{ cvt8lane_fp32_to_bf16(res) };
    }

    bfloat16x16 avg(bfloat16x16 arg0, bfloat16x16 arg1)
    {
        __m256i vlhs = arg0.data;
        __m256i vrhs = arg1.data;

        __m256 lhs_low = cvt8lane_bf16_to_fp32(detail::split_low(vlhs));
        __m256 lhs_high = cvt8lane_bf16_to_fp32(detail::split_high(vlhs));
        __m256 rhs_low = cvt8lane_bf16_to_fp32(detail::split_low(vrhs));
        __m256 rhs_high = cvt8lane_bf16_to_fp32(detail::split_high(vrhs));
        const __m256 vzero = _mm256_setzero_ps();
        const __m256 vhalf = _mm256_set1_ps(0.5f);
        __m256 res_low = _mm256_fmadd_ps(_mm256_add_ps(lhs_low, rhs_low), vhalf, vzero);
        __m256 res_high = _mm256_fmadd_ps(_mm256_add_ps(lhs_high, rhs_high), vhalf, vzero);

        return bfloat16x16{ detail::merge(
            cvt8lane_fp32_to_bf16(res_low),
            cvt8lane_fp32_to_bf16(res_high)) };
    }
#endif
#endif
}

namespace fyx::simd
{
    uint8x16 plus_sat(uint8x16 lhs, uint8x16 rhs) { return uint8x16{ _mm_adds_epu8(lhs.data, rhs.data) }; }
    uint16x8 plus_sat(uint16x8 lhs, uint16x8 rhs) { return uint16x8{ _mm_adds_epu16(lhs.data, rhs.data) }; }
    sint8x16 plus_sat(sint8x16 lhs, sint8x16 rhs) { return sint8x16{ _mm_adds_epi8(lhs.data, rhs.data) }; }
    sint16x8 plus_sat(sint16x8 lhs, sint16x8 rhs) { return sint16x8{ _mm_adds_epi16(lhs.data, rhs.data) }; }
    uint8x32 plus_sat(uint8x32 lhs, uint8x32 rhs) { return uint8x32{ _mm256_adds_epu8(lhs.data, rhs.data) }; }
    uint16x16 plus_sat(uint16x16 lhs, uint16x16 rhs) { return uint16x16{ _mm256_adds_epu16(lhs.data, rhs.data) }; }
    sint8x32 plus_sat(sint8x32 lhs, sint8x32 rhs) { return sint8x32{ _mm256_adds_epi8(lhs.data, rhs.data) }; }
    sint16x16 plus_sat(sint16x16 lhs, sint16x16 rhs) { return sint16x16{ _mm256_adds_epi16(lhs.data, rhs.data) }; }
    FOYE_SIMD_ERROR_WHEN_CALLED("No one would need such a feature") uint32x4 plus_sat(uint32x4 lhs, uint32x4 rhs) { return {}; }
    FOYE_SIMD_ERROR_WHEN_CALLED("No one would need such a feature") uint64x2 plus_sat(uint64x2 lhs, uint64x2 rhs) { return {}; }
    FOYE_SIMD_ERROR_WHEN_CALLED("No one would need such a feature") sint32x4 plus_sat(sint32x4 lhs, sint32x4 rhs) { return {}; }
    FOYE_SIMD_ERROR_WHEN_CALLED("No one would need such a feature") sint64x2 plus_sat(sint64x2 lhs, sint64x2 rhs) { return {}; }
    FOYE_SIMD_ERROR_WHEN_CALLED("No one would need such a feature") uint32x8 plus_sat(uint32x8 lhs, uint32x8 rhs) { return {}; }
    FOYE_SIMD_ERROR_WHEN_CALLED("No one would need such a feature") uint64x4 plus_sat(uint64x4 lhs, uint64x4 rhs) { return {}; }
    FOYE_SIMD_ERROR_WHEN_CALLED("No one would need such a feature") sint32x8 plus_sat(sint32x8 lhs, sint32x8 rhs) { return {}; }
    FOYE_SIMD_ERROR_WHEN_CALLED("No one would need such a feature") sint64x4 plus_sat(sint64x4 lhs, sint64x4 rhs) { return {}; }

    uint8x16 minus_sat(uint8x16 lhs, uint8x16 rhs) { return uint8x16{ _mm_subs_epu8(lhs.data, rhs.data) }; }
    uint16x8 minus_sat(uint16x8 lhs, uint16x8 rhs) { return uint16x8{ _mm_subs_epu16(lhs.data, rhs.data) }; }
    sint8x16 minus_sat(sint8x16 lhs, sint8x16 rhs) { return sint8x16{ _mm_subs_epi8(lhs.data, rhs.data) }; }
    sint16x8 minus_sat(sint16x8 lhs, sint16x8 rhs) { return sint16x8{ _mm_subs_epi16(lhs.data, rhs.data) }; }
    uint8x32 minus_sat(uint8x32 lhs, uint8x32 rhs) { return uint8x32{ _mm256_subs_epu8(lhs.data, rhs.data) }; }
    uint16x16 minus_sat(uint16x16 lhs, uint16x16 rhs) { return uint16x16{ _mm256_subs_epu16(lhs.data, rhs.data) }; }
    sint8x32 minus_sat(sint8x32 lhs, sint8x32 rhs) { return sint8x32{ _mm256_subs_epi8(lhs.data, rhs.data) }; }
    sint16x16 minus_sat(sint16x16 lhs, sint16x16 rhs) { return sint16x16{ _mm256_subs_epi16(lhs.data, rhs.data) }; }
    FOYE_SIMD_ERROR_WHEN_CALLED("No one would need such a feature") uint32x4 minus_sat(uint32x4 lhs, uint32x4 rhs) { return {}; }
    FOYE_SIMD_ERROR_WHEN_CALLED("No one would need such a feature") uint64x2 minus_sat(uint64x2 lhs, uint64x2 rhs) { return {}; }
    FOYE_SIMD_ERROR_WHEN_CALLED("No one would need such a feature") sint32x4 minus_sat(sint32x4 lhs, sint32x4 rhs) { return {}; }
    FOYE_SIMD_ERROR_WHEN_CALLED("No one would need such a feature") sint64x2 minus_sat(sint64x2 lhs, sint64x2 rhs) { return {}; }
    FOYE_SIMD_ERROR_WHEN_CALLED("No one would need such a feature") uint32x8 minus_sat(uint32x8 lhs, uint32x8 rhs) { return {}; }
    FOYE_SIMD_ERROR_WHEN_CALLED("No one would need such a feature") uint64x4 minus_sat(uint64x4 lhs, uint64x4 rhs) { return {}; }
    FOYE_SIMD_ERROR_WHEN_CALLED("No one would need such a feature") sint32x8 minus_sat(sint32x8 lhs, sint32x8 rhs) { return {}; }
    FOYE_SIMD_ERROR_WHEN_CALLED("No one would need such a feature") sint64x4 minus_sat(sint64x4 lhs, sint64x4 rhs) { return {}; }

    sint8x16 sign_transfer(sint8x16 data_source, sint8x16 sign_from) { return sint8x16{ _mm_sign_epi8(data_source.data, sign_from.data) }; }
    sint16x8 sign_transfer(sint16x8 data_source, sint16x8 sign_from) { return sint16x8{ _mm_sign_epi16(data_source.data, sign_from.data) }; }
    sint32x4 sign_transfer(sint32x4 data_source, sint32x4 sign_from) { return sint32x4{ _mm_sign_epi32(data_source.data, sign_from.data) }; }
    sint8x32 sign_transfer(sint8x32 data_source, sint8x32 sign_from) { return sint8x32{ _mm256_sign_epi8(data_source.data, sign_from.data) }; }
    sint16x16 sign_transfer(sint16x16 data_source, sint16x16 sign_from) { return sint16x16{ _mm256_sign_epi16(data_source.data, sign_from.data) }; }
    sint32x8 sign_transfer(sint32x8 data_source, sint32x8 sign_from) { return sint32x8{ _mm256_sign_epi32(data_source.data, sign_from.data) }; }

#if defined(_FOYE_SIMD_ENABLE_EMULATED_)
    sint64x2 sign_transfer(sint64x2 data_source, sint64x2 sign_from)
    {
        alignas(16) std::int64_t a_arr[2];
        alignas(16) std::int64_t b_arr[2];
        alignas(16) std::int64_t result_arr[2];

        store_aligned(data_source, a_arr);
        store_aligned(sign_from, b_arr);

        for (int i = 0; i < 2; i++) 
        {
            if (b_arr[i] > 0) 
            {
                result_arr[i] = a_arr[i];
            }
            else if (b_arr[i] == 0) 
            {
                result_arr[i] = 0;
            }
            else 
            {
                result_arr[i] = -a_arr[i];
            }
        }

        return load_aligned<sint64x2>(result_arr);
    }

    sint64x4 sign_transfer(sint64x4 data_source, sint64x4 sign_from)
    {
        __m256i a = data_source.data;
        __m256i b = sign_from.data;

        __m256i zero = _mm256_setzero_si256();

        __m256i gt_zero = _mm256_cmpgt_epi64(b, zero);
        __m256i eq_zero = _mm256_cmpeq_epi64(b, zero);

        __m256i neg_a = _mm256_sub_epi64(zero, a);

        __m256i result = _mm256_and_si256(gt_zero, a);
        result = _mm256_or_si256(result, neg_a);

        __m256i lt_zero = _mm256_andnot_si256(
            _mm256_or_si256(gt_zero, eq_zero), _mm256_set1_epi64x(-1LL));

        result = _mm256_and_si256(result, _mm256_or_si256(gt_zero, lt_zero));
        result = _mm256_andnot_si256(eq_zero, result);

        return sint64x4{ result };
    }
#endif

    int bitwise_test_zero(uint8x16 lhs, uint8x16 rhs) { return  _mm_testz_si128(lhs.data, rhs.data) ; }
    int bitwise_test_zero(uint16x8 lhs, uint16x8 rhs) { return  _mm_testz_si128(lhs.data, rhs.data) ; }
    int bitwise_test_zero(uint32x4 lhs, uint32x4 rhs) { return  _mm_testz_si128(lhs.data, rhs.data) ; }
    int bitwise_test_zero(uint64x2 lhs, uint64x2 rhs) { return  _mm_testz_si128(lhs.data, rhs.data) ; }
    int bitwise_test_zero(sint8x16 lhs, sint8x16 rhs) { return  _mm_testz_si128(lhs.data, rhs.data) ; }
    int bitwise_test_zero(sint16x8 lhs, sint16x8 rhs) { return  _mm_testz_si128(lhs.data, rhs.data) ; }
    int bitwise_test_zero(sint32x4 lhs, sint32x4 rhs) { return  _mm_testz_si128(lhs.data, rhs.data) ; }
    int bitwise_test_zero(sint64x2 lhs, sint64x2 rhs) { return  _mm_testz_si128(lhs.data, rhs.data) ; }
    int bitwise_test_zero(uint8x32 lhs, uint8x32 rhs) { return  _mm256_testz_si256(lhs.data, rhs.data) ; }
    int bitwise_test_zero(uint16x16 lhs, uint16x16 rhs) { return  _mm256_testz_si256(lhs.data, rhs.data) ; }
    int bitwise_test_zero(uint32x8 lhs, uint32x8 rhs) { return  _mm256_testz_si256(lhs.data, rhs.data) ; }
    int bitwise_test_zero(uint64x4 lhs, uint64x4 rhs) { return  _mm256_testz_si256(lhs.data, rhs.data) ; }
    int bitwise_test_zero(sint8x32 lhs, sint8x32 rhs) { return  _mm256_testz_si256(lhs.data, rhs.data) ; }
    int bitwise_test_zero(sint16x16 lhs, sint16x16 rhs) { return  _mm256_testz_si256(lhs.data, rhs.data) ; }
    int bitwise_test_zero(sint32x8 lhs, sint32x8 rhs) { return  _mm256_testz_si256(lhs.data, rhs.data) ; }
    int bitwise_test_zero(sint64x4 lhs, sint64x4 rhs) { return  _mm256_testz_si256(lhs.data, rhs.data) ; }
    int bitwise_test_zero(float32x8 lhs, float32x8 rhs) { return  _mm256_testz_ps(lhs.data, rhs.data) ; }
    int bitwise_test_zero(float32x4 lhs, float32x4 rhs) { return  _mm_testz_ps(lhs.data, rhs.data) ; }
    int bitwise_test_zero(float64x4 lhs, float64x4 rhs) { return  _mm256_testz_pd(lhs.data, rhs.data) ; }
    int bitwise_test_zero(float64x2 lhs, float64x2 rhs) { return  _mm_testz_pd(lhs.data, rhs.data) ; }

    int bitwise_test_not_zero(uint8x16 lhs, uint8x16 rhs) { return  _mm_testnzc_si128(lhs.data, rhs.data) ; }
    int bitwise_test_not_zero(uint16x8 lhs, uint16x8 rhs) { return  _mm_testnzc_si128(lhs.data, rhs.data) ; }
    int bitwise_test_not_zero(uint32x4 lhs, uint32x4 rhs) { return  _mm_testnzc_si128(lhs.data, rhs.data) ; }
    int bitwise_test_not_zero(uint64x2 lhs, uint64x2 rhs) { return  _mm_testnzc_si128(lhs.data, rhs.data) ; }
    int bitwise_test_not_zero(sint8x16 lhs, sint8x16 rhs) { return  _mm_testnzc_si128(lhs.data, rhs.data) ; }
    int bitwise_test_not_zero(sint16x8 lhs, sint16x8 rhs) { return  _mm_testnzc_si128(lhs.data, rhs.data) ; }
    int bitwise_test_not_zero(sint32x4 lhs, sint32x4 rhs) { return  _mm_testnzc_si128(lhs.data, rhs.data) ; }
    int bitwise_test_not_zero(sint64x2 lhs, sint64x2 rhs) { return  _mm_testnzc_si128(lhs.data, rhs.data) ; }
    int bitwise_test_not_zero(uint8x32 lhs, uint8x32 rhs) { return  _mm256_testnzc_si256(lhs.data, rhs.data) ; }
    int bitwise_test_not_zero(uint16x16 lhs, uint16x16 rhs) { return  _mm256_testnzc_si256(lhs.data, rhs.data) ; }
    int bitwise_test_not_zero(uint32x8 lhs, uint32x8 rhs) { return  _mm256_testnzc_si256(lhs.data, rhs.data) ; }
    int bitwise_test_not_zero(uint64x4 lhs, uint64x4 rhs) { return  _mm256_testnzc_si256(lhs.data, rhs.data) ; }
    int bitwise_test_not_zero(sint8x32 lhs, sint8x32 rhs) { return  _mm256_testnzc_si256(lhs.data, rhs.data) ; }
    int bitwise_test_not_zero(sint16x16 lhs, sint16x16 rhs) { return  _mm256_testnzc_si256(lhs.data, rhs.data) ; }
    int bitwise_test_not_zero(sint32x8 lhs, sint32x8 rhs) { return  _mm256_testnzc_si256(lhs.data, rhs.data) ; }
    int bitwise_test_not_zero(sint64x4 lhs, sint64x4 rhs) { return  _mm256_testnzc_si256(lhs.data, rhs.data) ; }
    int bitwise_test_not_zero(float32x8 lhs, float32x8 rhs) { return  _mm256_testnzc_ps(lhs.data, rhs.data) ; }
    int bitwise_test_not_zero(float32x4 lhs, float32x4 rhs) { return  _mm_testnzc_ps(lhs.data, rhs.data) ; }
    int bitwise_test_not_zero(float64x4 lhs, float64x4 rhs) { return  _mm256_testnzc_pd(lhs.data, rhs.data) ; }
    int bitwise_test_not_zero(float64x2 lhs, float64x2 rhs) { return  _mm_testnzc_pd(lhs.data, rhs.data) ; }

    int bitwise_test_check(uint8x16 lhs, uint8x16 rhs) { return  _mm_testc_si128(lhs.data, rhs.data) ; }
    int bitwise_test_check(uint16x8 lhs, uint16x8 rhs) { return  _mm_testc_si128(lhs.data, rhs.data) ; }
    int bitwise_test_check(uint32x4 lhs, uint32x4 rhs) { return  _mm_testc_si128(lhs.data, rhs.data) ; }
    int bitwise_test_check(uint64x2 lhs, uint64x2 rhs) { return  _mm_testc_si128(lhs.data, rhs.data) ; }
    int bitwise_test_check(sint8x16 lhs, sint8x16 rhs) { return  _mm_testc_si128(lhs.data, rhs.data) ; }
    int bitwise_test_check(sint16x8 lhs, sint16x8 rhs) { return  _mm_testc_si128(lhs.data, rhs.data) ; }
    int bitwise_test_check(sint32x4 lhs, sint32x4 rhs) { return  _mm_testc_si128(lhs.data, rhs.data) ; }
    int bitwise_test_check(sint64x2 lhs, sint64x2 rhs) { return  _mm_testc_si128(lhs.data, rhs.data) ; }
    int bitwise_test_check(uint8x32 lhs, uint8x32 rhs) { return  _mm256_testc_si256(lhs.data, rhs.data) ; }
    int bitwise_test_check(uint16x16 lhs, uint16x16 rhs) { return  _mm256_testc_si256(lhs.data, rhs.data) ; }
    int bitwise_test_check(uint32x8 lhs, uint32x8 rhs) { return  _mm256_testc_si256(lhs.data, rhs.data) ; }
    int bitwise_test_check(uint64x4 lhs, uint64x4 rhs) { return  _mm256_testc_si256(lhs.data, rhs.data) ; }
    int bitwise_test_check(sint8x32 lhs, sint8x32 rhs) { return  _mm256_testc_si256(lhs.data, rhs.data) ; }
    int bitwise_test_check(sint16x16 lhs, sint16x16 rhs) { return  _mm256_testc_si256(lhs.data, rhs.data) ; }
    int bitwise_test_check(sint32x8 lhs, sint32x8 rhs) { return  _mm256_testc_si256(lhs.data, rhs.data) ; }
    int bitwise_test_check(sint64x4 lhs, sint64x4 rhs) { return  _mm256_testc_si256(lhs.data, rhs.data) ; }
    int bitwise_test_check(float32x8 lhs, float32x8 rhs) { return  _mm256_testc_ps(lhs.data, rhs.data) ; }
    int bitwise_test_check(float32x4 lhs, float32x4 rhs) { return  _mm_testc_ps(lhs.data, rhs.data) ; }
    int bitwise_test_check(float64x4 lhs, float64x4 rhs) { return  _mm256_testc_pd(lhs.data, rhs.data) ; }
    int bitwise_test_check(float64x2 lhs, float64x2 rhs) { return  _mm_testc_pd(lhs.data, rhs.data) ; }


#if defined(_FOYE_SIMD_HAS_FP16_)
    int bitwise_test_zero(float16x8 lhs, float16x8 rhs) { return  _mm_testz_si128(lhs.data, rhs.data); }
    int bitwise_test_zero(float16x16 lhs, float16x16 rhs) { return  _mm256_testz_si256(lhs.data, rhs.data); }
    int bitwise_test_not_zero(float16x8 lhs, float16x8 rhs) { return  _mm_testnzc_si128(lhs.data, rhs.data); }
    int bitwise_test_not_zero(float16x16 lhs, float16x16 rhs) { return  _mm256_testnzc_si256(lhs.data, rhs.data); }
    int bitwise_test_check(float16x8 lhs, float16x8 rhs) { return  _mm_testc_si128(lhs.data, rhs.data); }
    int bitwise_test_check(float16x16 lhs, float16x16 rhs) { return  _mm256_testc_si256(lhs.data, rhs.data); }
#endif

#if defined(_FOYE_SIMD_HAS_BF16_)
    int bitwise_test_zero(bfloat16x8 lhs, bfloat16x8 rhs) { return  _mm_testz_si128(lhs.data, rhs.data); }
    int bitwise_test_zero(bfloat16x16 lhs, bfloat16x16 rhs) { return  _mm256_testz_si256(lhs.data, rhs.data); }
    int bitwise_test_not_zero(bfloat16x8 lhs, bfloat16x8 rhs) { return  _mm_testnzc_si128(lhs.data, rhs.data); }
    int bitwise_test_not_zero(bfloat16x16 lhs, bfloat16x16 rhs) { return  _mm256_testnzc_si256(lhs.data, rhs.data); }
    int bitwise_test_check(bfloat16x8 lhs, bfloat16x8 rhs) { return  _mm_testc_si128(lhs.data, rhs.data); }
    int bitwise_test_check(bfloat16x16 lhs, bfloat16x16 rhs) { return  _mm256_testc_si256(lhs.data, rhs.data); }
#endif
}

namespace fyx::simd
{
    template<typename simd_type>
    requires(is_basic_simd_v<simd_type>)
    simd_type& operator ++ (simd_type& input)
    {
        input = plus(input, load_brocast<simd_type>(1));
        return input;
    }

    template<typename simd_type>
    requires(is_basic_simd_v<simd_type>)
    simd_type operator ++ (simd_type& input, int)
    {
        simd_type temp = input;
        input = plus(input, load_brocast<simd_type>(1));
        return temp;
    }

    template<typename simd_type>
    requires(is_basic_simd_v<simd_type>)
    simd_type& operator -- (simd_type& input)
    {
        input = minus(input, load_brocast<simd_type>(1));
        return input;
    }

    template<typename simd_type>
    requires(is_basic_simd_v<simd_type>)
    simd_type operator -- (simd_type& input, int)
    {
        simd_type temp = input;
        input = minus(input, load_brocast<simd_type>(1));
        return temp;
    }

    template<typename simd_type>
    requires(is_basic_simd_v<simd_type>)
    simd_type operator - (simd_type input)
    {
        return negate(input);
    }

    template<typename simd_type>
    simd_type operator ~ (simd_type input)
    {
        return bitwise_NOT(input);
    }

#define _FOYE_SIMD_OPERATOR_DEFINE_(intrinsic_function, operator_symbol) \
template<typename simd_type> requires(is_basic_simd_v<simd_type>) \
simd_type operator operator_symbol (simd_type lhs, simd_type rhs)\
{\
    return intrinsic_function(lhs, rhs);\
}\
template<typename simd_type> requires(is_basic_simd_v<simd_type>)\
simd_type& operator operator_symbol##= (simd_type& lhs, simd_type rhs)\
{\
    lhs = intrinsic_function(lhs, rhs);\
    return lhs;\
}\
template<typename simd_type, typename rhs_type>\
    requires(is_basic_simd_v<simd_type>\
&& std::is_convertible_v<typename simd_type::scalar_t, rhs_type>)\
simd_type operator operator_symbol (simd_type lhs, rhs_type rhs)\
{\
    return intrinsic_function(lhs, load_brocast<simd_type>(rhs));\
}\
template<typename simd_type, typename rhs_type>\
    requires(is_basic_simd_v<simd_type>\
&& std::is_convertible_v<typename simd_type::scalar_t, rhs_type>)\
simd_type& operator operator_symbol##= (simd_type& lhs, rhs_type rhs)\
{\
    lhs = intrinsic_function(lhs, load_brocast<simd_type>(rhs));\
    return lhs;\
}

    _FOYE_SIMD_OPERATOR_DEFINE_(plus, +)
    _FOYE_SIMD_OPERATOR_DEFINE_(minus, -)
    _FOYE_SIMD_OPERATOR_DEFINE_(divide, /)
    _FOYE_SIMD_OPERATOR_DEFINE_(multiplies, *)

    _FOYE_SIMD_OPERATOR_DEFINE_(bitwise_AND, &)
    _FOYE_SIMD_OPERATOR_DEFINE_(bitwise_OR, |)
    _FOYE_SIMD_OPERATOR_DEFINE_(bitwise_XOR, ^)

    _FOYE_SIMD_OPERATOR_DEFINE_(shift_left, <<)
    _FOYE_SIMD_OPERATOR_DEFINE_(shift_right, >>)

        _FOYE_SIMD_OPERATOR_DEFINE_(remainder, %)

#undef _FOYE_SIMD_OPERATOR_DEFINE_
}




#undef DEF_NOTSUPPORTED_IMPLEMENT
#undef DEF_NOSUITABLE_IMPLEMENT
#endif
