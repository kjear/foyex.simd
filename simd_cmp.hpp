#ifndef _FOYE_SIMD_CMP_HPP_
#define _FOYE_SIMD_CMP_HPP_
#pragma once

#include "simd_def.hpp"
#include "simd_opt.hpp"

namespace fyx::simd
{
    mask_8x16 less(sint8x16 lhs, sint8x16 rhs) { return mask_8x16{ _mm_cmplt_epi8(lhs.data, rhs.data) }; }
    mask_16x8 less(sint16x8 lhs, sint16x8 rhs) { return mask_16x8{ _mm_cmplt_epi16(lhs.data, rhs.data) }; }
    mask_32x4 less(sint32x4 lhs, sint32x4 rhs) { return mask_32x4{ _mm_cmplt_epi32(lhs.data, rhs.data) }; }
    mask_64x2 less(sint64x2 lhs, sint64x2 rhs) { return mask_64x2{ _mm_cmpgt_epi64(rhs.data, lhs.data) }; }
    mask_8x32 less(sint8x32 lhs, sint8x32 rhs) { return mask_8x32{ _mm256_cmpgt_epi8(rhs.data, lhs.data) }; }
    mask_16x16 less(sint16x16 lhs, sint16x16 rhs) { return mask_16x16{ _mm256_cmpgt_epi16(rhs.data, lhs.data) }; }
    mask_32x8 less(sint32x8 lhs, sint32x8 rhs) { return mask_32x8{ _mm256_cmpgt_epi32(rhs.data, lhs.data) }; }
    mask_64x4 less(sint64x4 lhs, sint64x4 rhs) { return mask_64x4{ _mm256_cmpgt_epi64(rhs.data, lhs.data) }; }
    mask_32x4 less(float32x4 lhs, float32x4 rhs) { return mask_32x4{ _mm_cmplt_ps(lhs.data, rhs.data) }; }
    mask_32x8 less(float32x8 lhs, float32x8 rhs) { return mask_32x8{ _mm256_cmp_ps(lhs.data, rhs.data, _CMP_LT_OQ) }; }
    mask_64x2 less(float64x2 lhs, float64x2 rhs) { return mask_64x2{ _mm_cmplt_pd(lhs.data, rhs.data) }; }
    mask_64x4 less(float64x4 lhs, float64x4 rhs) { return mask_64x4{ _mm256_cmp_pd(lhs.data, rhs.data, _CMP_LT_OQ) }; }

    mask_8x16 greater(sint8x16 lhs, sint8x16 rhs) { return mask_8x16{ _mm_cmpgt_epi8(lhs.data, rhs.data) }; }
    mask_16x8 greater(sint16x8 lhs, sint16x8 rhs) { return mask_16x8{ _mm_cmpgt_epi16(lhs.data, rhs.data) }; }
    mask_32x4 greater(sint32x4 lhs, sint32x4 rhs) { return mask_32x4{ _mm_cmpgt_epi32(lhs.data, rhs.data) }; }
    mask_64x2 greater(sint64x2 lhs, sint64x2 rhs) { return mask_64x2{ _mm_cmpgt_epi64(lhs.data, rhs.data) }; }
    mask_8x32 greater(sint8x32 lhs, sint8x32 rhs) { return mask_8x32{ _mm256_cmpgt_epi8(lhs.data, rhs.data) }; }
    mask_16x16 greater(sint16x16 lhs, sint16x16 rhs) { return mask_16x16{ _mm256_cmpgt_epi16(lhs.data, rhs.data) }; }
    mask_32x8 greater(sint32x8 lhs, sint32x8 rhs) { return mask_32x8{ _mm256_cmpgt_epi32(lhs.data, rhs.data) }; }
    mask_64x4 greater(sint64x4 lhs, sint64x4 rhs) { return mask_64x4{ _mm256_cmpgt_epi64(lhs.data, rhs.data) }; }
    mask_32x4 greater(float32x4 lhs, float32x4 rhs) { return mask_32x4{ _mm_cmpgt_ps(lhs.data, rhs.data) }; }
    mask_32x8 greater(float32x8 lhs, float32x8 rhs) { return mask_32x8{ _mm256_cmp_ps(lhs.data, rhs.data, _CMP_GT_OQ) }; }
    mask_64x2 greater(float64x2 lhs, float64x2 rhs) { return mask_64x2{ _mm_cmpgt_pd(lhs.data, rhs.data) }; }
    mask_64x4 greater(float64x4 lhs, float64x4 rhs) { return mask_64x4{ _mm256_cmp_pd(lhs.data, rhs.data, _CMP_GT_OQ) }; }


    mask_8x16 equal(sint8x16 lhs, sint8x16 rhs) { return mask_8x16{ _mm_cmpeq_epi8(lhs.data, rhs.data) }; }
    mask_16x8 equal(sint16x8 lhs, sint16x8 rhs) { return mask_16x8{ _mm_cmpeq_epi16(lhs.data, rhs.data) }; }
    mask_32x4 equal(sint32x4 lhs, sint32x4 rhs) { return mask_32x4{ _mm_cmpeq_epi32(lhs.data, rhs.data) }; }
    mask_64x2 equal(sint64x2 lhs, sint64x2 rhs) { return mask_64x2{ _mm_cmpeq_epi64(lhs.data, rhs.data) }; }
    mask_8x16 equal(uint8x16 lhs, uint8x16 rhs) { return mask_8x16{ _mm_cmpeq_epi8(lhs.data, rhs.data) }; }
    mask_16x8 equal(uint16x8 lhs, uint16x8 rhs) { return mask_16x8{ _mm_cmpeq_epi16(lhs.data, rhs.data) }; }
    mask_32x4 equal(uint32x4 lhs, uint32x4 rhs) { return mask_32x4{ _mm_cmpeq_epi32(lhs.data, rhs.data) }; }
    mask_64x2 equal(uint64x2 lhs, uint64x2 rhs) { return mask_64x2{ _mm_cmpeq_epi64(lhs.data, rhs.data) }; }
    mask_8x32 equal(sint8x32 lhs, sint8x32 rhs) { return mask_8x32{ _mm256_cmpeq_epi8(lhs.data, rhs.data) }; }
    mask_16x16 equal(sint16x16 lhs, sint16x16 rhs) { return mask_16x16{ _mm256_cmpeq_epi16(lhs.data, rhs.data) }; }
    mask_32x8 equal(sint32x8 lhs, sint32x8 rhs) { return mask_32x8{ _mm256_cmpeq_epi32(lhs.data, rhs.data) }; }
    mask_64x4 equal(sint64x4 lhs, sint64x4 rhs) { return mask_64x4{ _mm256_cmpeq_epi64(lhs.data, rhs.data) }; }
    mask_8x32 equal(uint8x32 lhs, uint8x32 rhs) { return mask_8x32{ _mm256_cmpeq_epi8(lhs.data, rhs.data) }; }
    mask_16x16 equal(uint16x16 lhs, uint16x16 rhs) { return mask_16x16{ _mm256_cmpeq_epi16(lhs.data, rhs.data) }; }
    mask_32x8 equal(uint32x8 lhs, uint32x8 rhs) { return mask_32x8{ _mm256_cmpeq_epi32(lhs.data, rhs.data) }; }
    mask_64x4 equal(uint64x4 lhs, uint64x4 rhs) { return mask_64x4{ _mm256_cmpeq_epi64(lhs.data, rhs.data) }; }

    mask_8x16 equal(uint8x16 lhs, sint8x16 rhs) { return mask_8x16{ _mm_cmpeq_epi8(lhs.data, rhs.data) }; }
    mask_16x8 equal(uint16x8 lhs, sint16x8 rhs) { return mask_16x8{ _mm_cmpeq_epi16(lhs.data, rhs.data) }; }
    mask_32x4 equal(uint32x4 lhs, sint32x4 rhs) { return mask_32x4{ _mm_cmpeq_epi32(lhs.data, rhs.data) }; }
    mask_64x2 equal(uint64x2 lhs, sint64x2 rhs) { return mask_64x2{ _mm_cmpeq_epi64(lhs.data, rhs.data) }; }
    mask_8x32 equal(uint8x32 lhs, sint8x32 rhs) { return mask_8x32{ _mm256_cmpeq_epi8(lhs.data, rhs.data) }; }
    mask_16x16 equal(uint16x16 lhs, sint16x16 rhs) { return mask_16x16{ _mm256_cmpeq_epi16(lhs.data, rhs.data) }; }
    mask_32x8 equal(uint32x8 lhs, sint32x8 rhs) { return mask_32x8{ _mm256_cmpeq_epi32(lhs.data, rhs.data) }; }
    mask_64x4 equal(uint64x4 lhs, sint64x4 rhs) { return mask_64x4{ _mm256_cmpeq_epi64(lhs.data, rhs.data) }; }
    
    mask_8x16 equal(sint8x16 lhs, uint8x16 rhs) { return mask_8x16{ _mm_cmpeq_epi8(lhs.data, rhs.data) }; }
    mask_16x8 equal(sint16x8 lhs, uint16x8 rhs) { return mask_16x8{ _mm_cmpeq_epi16(lhs.data, rhs.data) }; }
    mask_32x4 equal(sint32x4 lhs, uint32x4 rhs) { return mask_32x4{ _mm_cmpeq_epi32(lhs.data, rhs.data) }; }
    mask_64x2 equal(sint64x2 lhs, uint64x2 rhs) { return mask_64x2{ _mm_cmpeq_epi64(lhs.data, rhs.data) }; }
    mask_8x32 equal(sint8x32 lhs, uint8x32 rhs) { return mask_8x32{ _mm256_cmpeq_epi8(lhs.data, rhs.data) }; }
    mask_16x16 equal(sint16x16 lhs, uint16x16 rhs) { return mask_16x16{ _mm256_cmpeq_epi16(lhs.data, rhs.data) }; }
    mask_32x8 equal(sint32x8 lhs, uint32x8 rhs) { return mask_32x8{ _mm256_cmpeq_epi32(lhs.data, rhs.data) }; }
    mask_64x4 equal(sint64x4 lhs, uint64x4 rhs) { return mask_64x4{ _mm256_cmpeq_epi64(lhs.data, rhs.data) }; }

    mask_32x4 equal(float32x4 lhs, float32x4 rhs) { return mask_32x4{ _mm_cmpeq_ps(rhs.data, lhs.data) }; }
    mask_32x8 equal(float32x8 lhs, float32x8 rhs) { return mask_32x8{ _mm256_cmp_ps(rhs.data, lhs.data, _CMP_EQ_OQ) }; }
    mask_64x2 equal(float64x2 lhs, float64x2 rhs) { return mask_64x2{ _mm_cmpeq_pd(rhs.data, lhs.data) }; }
    mask_64x4 equal(float64x4 lhs, float64x4 rhs) { return mask_64x4{ _mm256_cmp_pd(rhs.data, lhs.data, _CMP_EQ_OQ) }; }

    template<typename simd_lhs_type, typename simd_rhs_type> 
    requires(is_basic_simd_v<simd_lhs_type> && is_basic_simd_v<simd_rhs_type>
    && simd_lhs_type::bit_width == simd_rhs_type::bit_width
        && simd_lhs_type::scalar_bit_width == simd_rhs_type::scalar_bit_width)
    mask_from_simd_t<simd_lhs_type> not_equal(simd_lhs_type lhs, simd_rhs_type rhs)
    {
        return mask_from_simd_t<simd_lhs_type>{ fyx::simd::bitwise_NOT(simd_lhs_type{ fyx::simd::equal(lhs, rhs) }).data };
    }

    template<typename simd_type> requires(fyx::simd::is_basic_simd_v<simd_type>)
    mask_from_simd_t<simd_type> less_equal(simd_type lhs, simd_type rhs)
    {
        return mask_from_simd_t<simd_type>{
            fyx::simd::bitwise_OR(
                simd_type{ fyx::simd::less(rhs, lhs) },
                simd_type{ fyx::simd::equal(rhs, lhs) }).data };
    }

    template<typename simd_type> requires(fyx::simd::is_basic_simd_v<simd_type>)
    mask_from_simd_t<simd_type> greater_equal(simd_type lhs, simd_type rhs)
    {
        return mask_from_simd_t<simd_type>{
            fyx::simd::bitwise_OR(
                simd_type{ fyx::simd::greater(rhs, lhs) },
                simd_type{ fyx::simd::equal(rhs, lhs) }).data };
    }

    template<> mask_32x4 not_equal(float32x4 lhs, float32x4 rhs) { return mask_32x4{ _mm_cmpneq_ps(rhs.data, lhs.data) }; }
    template<> mask_32x8 not_equal(float32x8 lhs, float32x8 rhs) { return mask_32x8{ _mm256_cmp_ps(rhs.data, lhs.data, _CMP_NEQ_OQ) }; }
    template<> mask_64x2 not_equal(float64x2 lhs, float64x2 rhs) { return mask_64x2{ _mm_cmpneq_pd(rhs.data, lhs.data) }; }
    template<> mask_64x4 not_equal(float64x4 lhs, float64x4 rhs) { return mask_64x4{ _mm256_cmp_pd(rhs.data, lhs.data, _CMP_NEQ_OQ) }; }

    template<> mask_32x4 less_equal(float32x4 lhs, float32x4 rhs) { return mask_32x4{ _mm_cmple_ps(rhs.data, lhs.data) }; }
    template<> mask_32x8 less_equal(float32x8 lhs, float32x8 rhs) { return mask_32x8{ _mm256_cmp_ps(rhs.data, lhs.data, _CMP_LE_OQ) }; }
    template<> mask_64x2 less_equal(float64x2 lhs, float64x2 rhs) { return mask_64x2{ _mm_cmple_pd(rhs.data, lhs.data) }; }
    template<> mask_64x4 less_equal(float64x4 lhs, float64x4 rhs) { return mask_64x4{ _mm256_cmp_pd(rhs.data, lhs.data, _CMP_LE_OQ) }; }

    template<> mask_32x4 greater_equal(float32x4 lhs, float32x4 rhs) { return mask_32x4{ _mm_cmpge_ps(rhs.data, lhs.data) }; }
    template<> mask_32x8 greater_equal(float32x8 lhs, float32x8 rhs) { return mask_32x8{ _mm256_cmp_ps(rhs.data, lhs.data, _CMP_GE_OQ) }; }
    template<> mask_64x2 greater_equal(float64x2 lhs, float64x2 rhs) { return mask_64x2{ _mm_cmpge_pd(rhs.data, lhs.data) }; }
    template<> mask_64x4 greater_equal(float64x4 lhs, float64x4 rhs) { return mask_64x4{ _mm256_cmp_pd(rhs.data, lhs.data, _CMP_GE_OQ) }; }

#if defined(_FOYE_SIMD_HAS_FP16_) || defined(_FOYE_SIMD_HAS_BF16_)
#define _FOYE_SIMD_CMP_LESS_(lhs, rhs) (_mm256_cmp_ps(lhs, rhs, _CMP_LT_OQ))
#define _FOYE_SIMD_CMP_GREATER_(lhs, rhs) (_mm256_cmp_ps(lhs, rhs, _CMP_GT_OQ))
#define _FOYE_SIMD_CMP_EQUAL_(lhs, rhs) (_mm256_cmp_ps(lhs, rhs, _CMP_EQ_OQ))
#define _FOYE_SIMD_CMP_LESS_EQUAL_(lhs, rhs) (_mm256_cmp_ps(lhs, rhs, _CMP_LE_OQ))
#define _FOYE_SIMD_CMP_GREATER_EQUAL_(lhs, rhs) (_mm256_cmp_ps(lhs, rhs, _CMP_GE_OQ))
#define _FOYE_SIMD_CMP_NOT_EQUAL_(lhs, rhs) (_mm256_cmp_ps(lhs, rhs, _CMP_NEQ_OQ))
#define _FOYE_SIMD_DEFINE_HALF_CMP_FUNCION_(funcname, vhalf_type, s_to_half, half_to_s, cmp_expr) \
mask_16x8 funcname(vhalf_type##x8 lhs, vhalf_type##x8 rhs) \
{\
    return mask_16x8{ s_to_half(cmp_expr(\
        half_to_s(lhs.data),\
        half_to_s(rhs.data))) };\
}\
mask_16x16 funcname(vhalf_type##x16 lhs, vhalf_type##x16 rhs)\
{\
    __m256 res_low = cmp_expr(half_to_s(detail::split_low(lhs.data)),\
        half_to_s(detail::split_low(rhs.data)));\
    __m256 res_high = cmp_expr(half_to_s(detail::split_high(lhs.data)),\
        half_to_s(detail::split_high(rhs.data)));\
    __m256i half_res = detail::merge(s_to_half(res_low),\
        s_to_half(res_high));\
    return mask_16x16{ half_res };\
}

#define _FOYE_SIMD_DEFINE_HALF_MIX_CMP_FUNCION_(funcname, vhalf_type, s_to_half, half_to_s, cmp_expr) \
template<> mask_16x8 funcname(vhalf_type##x8 lhs, vhalf_type##x8 rhs) \
{\
    return mask_16x8{ s_to_half(cmp_expr(\
        half_to_s(lhs.data),\
        half_to_s(rhs.data))) };\
}\
template<> mask_16x16 funcname(vhalf_type##x16 lhs, vhalf_type##x16 rhs)\
{\
    __m256 res_low = cmp_expr(half_to_s(detail::split_low(lhs.data)),\
        half_to_s(detail::split_low(rhs.data)));\
    __m256 res_high = cmp_expr(half_to_s(detail::split_high(lhs.data)),\
        half_to_s(detail::split_high(rhs.data)));\
    __m256i half_res = detail::merge(s_to_half(res_low),\
        s_to_half(res_high));\
    return mask_16x16{ half_res };\
}
    
#else
#define _FOYE_SIMD_CMP_LESS_(...)
#define _FOYE_SIMD_CMP_GREATER_(...)
#define _FOYE_SIMD_CMP_EQUAL_(...)
#define _FOYE_SIMD_CMP_LESS_EQUAL_(...)
#define _FOYE_SIMD_CMP_GREATER_EQUAL_(...)
#define _FOYE_SIMD_CMP_NOT_EQUAL_(...)
#define _FOYE_SIMD_DEFINE_HALF_CMP_FUNCION_(...)
#define _FOYE_SIMD_DEFINE_HALF_MIX_CMP_FUNCION_(...)
#endif

#if defined(_FOYE_SIMD_HAS_FP16_)
    _FOYE_SIMD_DEFINE_HALF_CMP_FUNCION_(less, float16, cvt8lane_fp32_to_fp16, cvt8lane_fp16_to_fp32, _FOYE_SIMD_CMP_LESS_)
    _FOYE_SIMD_DEFINE_HALF_CMP_FUNCION_(greater, float16, cvt8lane_fp32_to_fp16, cvt8lane_fp16_to_fp32, _FOYE_SIMD_CMP_GREATER_)
    _FOYE_SIMD_DEFINE_HALF_CMP_FUNCION_(equal, float16, cvt8lane_fp32_to_fp16, cvt8lane_fp16_to_fp32, _FOYE_SIMD_CMP_EQUAL_)
    _FOYE_SIMD_DEFINE_HALF_MIX_CMP_FUNCION_(not_equal, float16, cvt8lane_fp32_to_fp16, cvt8lane_fp16_to_fp32, _FOYE_SIMD_CMP_NOT_EQUAL_)
    _FOYE_SIMD_DEFINE_HALF_MIX_CMP_FUNCION_(less_equal, float16, cvt8lane_fp32_to_fp16, cvt8lane_fp16_to_fp32, _FOYE_SIMD_CMP_LESS_EQUAL_)
    _FOYE_SIMD_DEFINE_HALF_MIX_CMP_FUNCION_(greater_equal, float16, cvt8lane_fp32_to_fp16, cvt8lane_fp16_to_fp32, _FOYE_SIMD_CMP_GREATER_EQUAL_)
#endif
#if defined(_FOYE_SIMD_HAS_BF16_)
    _FOYE_SIMD_DEFINE_HALF_CMP_FUNCION_(less, bfloat16, cvt8lane_fp32_to_bf16, cvt8lane_bf16_to_fp32, _FOYE_SIMD_CMP_LESS_)
    _FOYE_SIMD_DEFINE_HALF_CMP_FUNCION_(greater, bfloat16, cvt8lane_fp32_to_bf16, cvt8lane_bf16_to_fp32, _FOYE_SIMD_CMP_GREATER_)
    _FOYE_SIMD_DEFINE_HALF_CMP_FUNCION_(equal, bfloat16, cvt8lane_fp32_to_bf16, cvt8lane_bf16_to_fp32, _FOYE_SIMD_CMP_EQUAL_)
    _FOYE_SIMD_DEFINE_HALF_MIX_CMP_FUNCION_(not_equal, bfloat16, cvt8lane_fp32_to_bf16, cvt8lane_bf16_to_fp32, _FOYE_SIMD_CMP_NOT_EQUAL_)
    _FOYE_SIMD_DEFINE_HALF_MIX_CMP_FUNCION_(less_equal, bfloat16, cvt8lane_fp32_to_bf16, cvt8lane_bf16_to_fp32, _FOYE_SIMD_CMP_LESS_EQUAL_)
    _FOYE_SIMD_DEFINE_HALF_MIX_CMP_FUNCION_(greater_equal, bfloat16, cvt8lane_fp32_to_bf16, cvt8lane_bf16_to_fp32, _FOYE_SIMD_CMP_GREATER_EQUAL_)
#endif


#if !defined(_FOYE_SIMD_ENABLE_EMULATED_)
    DEF_NOTSUPPORTED_IMPLEMENT(uint8x16 less(uint8x16 lhs, uint8x16 rhs))
    DEF_NOTSUPPORTED_IMPLEMENT(uint16x8 less(uint16x8 lhs, uint16x8 rhs))
    DEF_NOTSUPPORTED_IMPLEMENT(uint32x4 less(uint32x4 lhs, uint32x4 rhs))
    DEF_NOTSUPPORTED_IMPLEMENT(uint64x2 less(uint64x2 lhs, uint64x2 rhs))
    DEF_NOTSUPPORTED_IMPLEMENT(uint8x32 less(uint8x32 lhs, uint8x32 rhs))
    DEF_NOTSUPPORTED_IMPLEMENT(uint16x16 less(uint16x16 lhs, uint16x16 rhs))
    DEF_NOTSUPPORTED_IMPLEMENT(uint32x8 less(uint32x8 lhs, uint32x8 rhs))
    DEF_NOTSUPPORTED_IMPLEMENT(uint64x4 less(uint64x4 lhs, uint64x4 rhs))

    DEF_NOTSUPPORTED_IMPLEMENT(uint8x16 greater(uint8x16 lhs, uint8x16 rhs))
    DEF_NOTSUPPORTED_IMPLEMENT(uint16x8 greater(uint16x8 lhs, uint16x8 rhs))
    DEF_NOTSUPPORTED_IMPLEMENT(uint32x4 greater(uint32x4 lhs, uint32x4 rhs))
    DEF_NOTSUPPORTED_IMPLEMENT(uint64x2 greater(uint64x2 lhs, uint64x2 rhs))
    DEF_NOTSUPPORTED_IMPLEMENT(uint8x32 greater(uint8x32 lhs, uint8x32 rhs))
    DEF_NOTSUPPORTED_IMPLEMENT(uint16x16 greater(uint16x16 lhs, uint16x16 rhs))
    DEF_NOTSUPPORTED_IMPLEMENT(uint32x8 greater(uint32x8 lhs, uint32x8 rhs))
    DEF_NOTSUPPORTED_IMPLEMENT(uint64x4 greater(uint64x4 lhs, uint64x4 rhs))
#else
#define DEFINE_LESSGREATER_UNSIGNED_VERSION(funcname, unsigned_simd_type, expr)\
basic_mask<unsigned_simd_type::lane_width, unsigned_simd_type::bit_width> funcname(unsigned_simd_type lhs, unsigned_simd_type rhs)\
{\
    using source_scalar_type = typename unsigned_simd_type::scalar_t;\
    using vector_type = typename unsigned_simd_type::vector_t;\
    using signed_scalar_type = std::make_signed_t<source_scalar_type>;\
    using signed_simd_type = basic_simd<signed_scalar_type, unsigned_simd_type::bit_width>;\
    constexpr signed_scalar_type shift_amount = static_cast<signed_scalar_type>(1) << (sizeof(source_scalar_type) * CHAR_BIT - 1);\
    const unsigned_simd_type flip_bit{ fyx::simd::detail::brocast<vector_type>(static_cast<source_scalar_type>(shift_amount)) };\
    unsigned_simd_type fleft = fyx::simd::bitwise_XOR(lhs, flip_bit);\
    unsigned_simd_type fright = fyx::simd::bitwise_XOR(rhs, flip_bit);\
    signed_simd_type signed_left = fyx::simd::reinterpret<signed_simd_type>(fleft);\
    signed_simd_type signed_right = fyx::simd::reinterpret<signed_simd_type>(fright);\
    return expr(signed_left, signed_right);\
}
#define DEFINE_LESSGREATER_UNSIGNED_VERSION_2WAY_DISPATCH(input_vector_type) \
    DEFINE_LESSGREATER_UNSIGNED_VERSION(less, input_vector_type, fyx::simd::less) \
    DEFINE_LESSGREATER_UNSIGNED_VERSION(greater, input_vector_type, fyx::simd::greater)

    DEFINE_LESSGREATER_UNSIGNED_VERSION_2WAY_DISPATCH(uint8x16)
    DEFINE_LESSGREATER_UNSIGNED_VERSION_2WAY_DISPATCH(uint16x8)
    DEFINE_LESSGREATER_UNSIGNED_VERSION_2WAY_DISPATCH(uint32x4)
    DEFINE_LESSGREATER_UNSIGNED_VERSION_2WAY_DISPATCH(uint64x2)
    DEFINE_LESSGREATER_UNSIGNED_VERSION_2WAY_DISPATCH(uint8x32)
    DEFINE_LESSGREATER_UNSIGNED_VERSION_2WAY_DISPATCH(uint16x16)
    DEFINE_LESSGREATER_UNSIGNED_VERSION_2WAY_DISPATCH(uint32x8)
    DEFINE_LESSGREATER_UNSIGNED_VERSION_2WAY_DISPATCH(uint64x4)
#undef DEFINE_LESSGREATER_UNSIGNED_VERSION_2WAY_DISPATCH
#undef DEFINE_LESSGREATER_UNSIGNED_VERSION
#endif
}

#if defined(FOYE_SIMD_ENABLE_COMPARISON_OPERATORS)
namespace fyx::simd
{
    template<typename simd_lhs_type, typename simd_rhs_type>
        requires(is_basic_simd_v<simd_lhs_type>&& is_basic_simd_v<simd_rhs_type>
    && simd_lhs_type::bit_width == simd_rhs_type::bit_width
        && simd_lhs_type::scalar_bit_width == simd_rhs_type::scalar_bit_width)
    mask_from_simd_t<simd_lhs_type> operator == (simd_lhs_type lhs, simd_rhs_type rhs)
    {
        return equal(lhs, rhs);
    }

    template<typename simd_lhs_type, typename simd_rhs_type>
        requires(is_basic_simd_v<simd_lhs_type>&& is_basic_simd_v<simd_rhs_type>
    && simd_lhs_type::bit_width == simd_rhs_type::bit_width
        && simd_lhs_type::scalar_bit_width == simd_rhs_type::scalar_bit_width)
    mask_from_simd_t<simd_lhs_type> operator != (simd_lhs_type lhs, simd_rhs_type rhs)
    {
        return bitwise_NOT(equal(lhs, rhs).as_basic_simd<simd_lhs_type>()).as_basic_mask();
    }

    template<typename simd_type> requires(is_basic_simd_v<simd_type>)
    mask_from_simd_t<simd_type> operator > (simd_type lhs, simd_type rhs)
    {
        return greater(lhs, rhs);
    }

    template<typename simd_type> requires(is_basic_simd_v<simd_type>)
    mask_from_simd_t<simd_type> operator < (simd_type lhs, simd_type rhs)
    {
        return less(lhs, rhs);
    }

    template<typename simd_type> requires(is_basic_simd_v<simd_type>)
    mask_from_simd_t<simd_type> operator >= (simd_type lhs, simd_type rhs)
    {
        return greater_equal(lhs, rhs);
    }

    template<typename simd_type> requires(is_basic_simd_v<simd_type>)
    mask_from_simd_t<simd_type> operator <= (simd_type lhs, simd_type rhs)
    {
        return less_equal(lhs, rhs);
    }

    template<typename simd_type, typename rhs_type> requires(is_basic_simd_v<simd_type>
    && std::is_constructible_v<typename simd_type::scalar_t, rhs_type>)
    mask_from_simd_t<simd_type> operator == (simd_type lhs, const rhs_type& rhs)
    {
        simd_type rhs_vec = load_brocast<simd_type>(typename simd_type::scalar_t(rhs));
        return equal(lhs, rhs_vec);
    }

    template<typename simd_type, typename rhs_type> requires(is_basic_simd_v<simd_type>
    && std::is_constructible_v<typename simd_type::scalar_t, rhs_type>)
    mask_from_simd_t<simd_type> operator != (simd_type lhs, const rhs_type& rhs)
    {
        simd_type rhs_vec = load_brocast<simd_type>(typename simd_type::scalar_t(rhs));
        return bitwise_NOT(equal(lhs, rhs_vec).as_basic_simd<simd_type>()).as_basic_mask();
    }

    template<typename simd_type, typename rhs_type> requires(is_basic_simd_v<simd_type>
    && std::is_constructible_v<typename simd_type::scalar_t, rhs_type>)
    mask_from_simd_t<simd_type> operator > (simd_type lhs, const rhs_type& rhs)
    {
        simd_type rhs_vec = load_brocast<simd_type>(typename simd_type::scalar_t(rhs));
        return greater(lhs, rhs_vec);
    }

    template<typename simd_type, typename rhs_type> requires(is_basic_simd_v<simd_type>
    && std::is_constructible_v<typename simd_type::scalar_t, rhs_type>)
    mask_from_simd_t<simd_type> operator < (simd_type lhs, const rhs_type& rhs)
    {
        simd_type rhs_vec = load_brocast<simd_type>(typename simd_type::scalar_t(rhs));
        return less(lhs, rhs_vec);
    }

    template<typename simd_type, typename rhs_type> requires(is_basic_simd_v<simd_type>
    && std::is_constructible_v<typename simd_type::scalar_t, rhs_type>)
    mask_from_simd_t<simd_type> operator >= (simd_type lhs, const rhs_type& rhs)
    {
        simd_type rhs_vec = load_brocast<simd_type>(typename simd_type::scalar_t(rhs));
        return greater_equal(lhs, rhs_vec);
    }

    template<typename simd_type, typename rhs_type> requires(is_basic_simd_v<simd_type>
    && std::is_constructible_v<typename simd_type::scalar_t, rhs_type>)
    mask_from_simd_t<simd_type> operator <= (simd_type lhs, const rhs_type& rhs)
    {
        simd_type rhs_vec = load_brocast<simd_type>(typename simd_type::scalar_t(rhs));
        return less_equal(lhs, rhs_vec);
    }
}
#endif

#endif
