#ifndef _FOYE_SIMD_MASK_HPP_
#define _FOYE_SIMD_MASK_HPP_
#pragma once

#include "simd_def.hpp"
#include "simd_utility.hpp"
#include "simd_opt.hpp"
#include "simd_cmp.hpp"
#include "simd_interleave.hpp"

namespace fyx::simd
{
    template<typename simd_type> requires(fyx::simd::is_basic_simd_v<simd_type>)
    mask_from_simd_t<simd_type> where_between(simd_type value, simd_type low, simd_type high)
    {
        const simd_type clamped = fyx::simd::max(low, fyx::simd::min(value, high));
        return fyx::simd::equal(value, clamped);
    }
    
    template<typename simd_type> requires(fyx::simd::is_basic_simd_v<simd_type>)
    mask_from_simd_t<simd_type> where_positive(simd_type value)
    {
        const simd_type zero = fyx::simd::allzero_bits_as<simd_type>();
        return fyx::simd::greater(value, zero);
    }

    template<typename simd_type> requires(fyx::simd::is_basic_simd_v<simd_type>)
    mask_from_simd_t<simd_type> where_negative(simd_type value)
    {
        if constexpr (std::is_unsigned_v<typename simd_type::scalar_t>)
        {
            return mask_from_simd_t<simd_type>{ fyx::simd::allzero_bits_as<simd_type>() };
        }
        else
        {
            const simd_type zero = fyx::simd::allzero_bits_as<simd_type>();
            return fyx::simd::greater(zero, value);
        }
    }

    template<typename simd_type> requires(is_basic_simd_v<simd_type>)
    mask_from_simd_t<simd_type> where_in_range(simd_type value,
            typename simd_type::scalar_t low, typename simd_type::scalar_t high) 
    {
        return where_between(value, load_brocast<simd_type>(low), load_brocast<simd_type>(high));
    }

    template<int input0_first, int input0_second, int input1_first, int input1_second>
    requires((input0_first >= 0 && input0_first <= 3) && (input0_second >= 0 && input0_second <= 3)
    && (input1_first >= 0 && input1_first <= 3) && (input1_second >= 0 && input1_second <= 3))
    float32x4 shuffle(float32x4 input_0, float32x4 input_1)
    {
        return float32x4{ _mm_shuffle_ps(input_0.data, input_1.data,
            (((input0_first) << 6) | ((input0_second) << 4) |
                ((input1_first) << 2) | ((input1_second)))) };
    }

    template<int input_selected, int input1_selected>
    requires((input_selected == 1 || input_selected == 0) && 
    (input1_selected == 1 || input1_selected == 0))
    float64x2 shuffle(float64x2 input_0, float64x2 input_1)
    {
        return float64x2{ _mm_shuffle_pd(input_0.data, input_1.data,
            (((input_selected) << 1) | (input1_selected))) };
    }

    namespace detail
    {
        template<typename mask_type, bool first> requires(fyx::simd::is_basic_mask_v<mask_type>)
        int where_target_impl(mask_type source)
        {
            int mask{ 0 };
            if constexpr (fyx::simd::is_128bits_mask_v<mask_type>)
            {
                __m128i value = source.data;
                if constexpr (mask_type::lane_width == 2) { mask = _mm_movemask_pd(_mm_castsi128_pd(value)); }
                else if constexpr (mask_type::lane_width == 4) { mask = _mm_movemask_ps(_mm_castsi128_ps(value)); }
                else if constexpr (mask_type::lane_width == 8) { mask = _mm_movemask_epi8(_mm_packs_epi16(value, value)); }
                else if constexpr (mask_type::lane_width == 16) { mask = _mm_movemask_epi8(value); }
                else { __assume(false); }
            }
            else if constexpr (fyx::simd::is_256bits_mask_v<mask_type>)
            {
                __m256i value = source.data;
                if constexpr (mask_type::lane_width == 4) { mask = _mm256_movemask_pd(_mm256_castsi256_pd(value)); }
                else if constexpr (mask_type::lane_width == 8) { mask = _mm256_movemask_ps(_mm256_castsi256_ps(value)); }
                else if constexpr (mask_type::lane_width == 16) { mask = _mm256_movemask_epi8(_mm256_packs_epi16(value, value)); }
                else if constexpr (mask_type::lane_width == 32) { mask = _mm256_movemask_epi8(value); }
                else { __assume(false); }
            }
            else
            {
                __assume(false);
            }

            unsigned long index;
            if constexpr (first)
            {
                _BitScanForward(&index, mask);
            }
            else
            {
                _BitScanReverse(&index, mask);
            }
            return static_cast<int>(index);
        }
    }

    template<typename mask_type> requires(fyx::simd::is_basic_mask_v<mask_type>)
    int where_first(mask_type source) { return fyx::simd::detail::where_target_impl<mask_type, true>(source); }

    template<typename mask_type> requires(fyx::simd::is_basic_mask_v<mask_type>)
    int where_last(mask_type source) { return fyx::simd::detail::where_target_impl<mask_type, false>(source); }

    template<typename simd_type> requires(fyx::simd::is_basic_simd_v<simd_type>)
    simd_type where_assign(simd_type source, simd_type value_to_assign, fyx::simd::mask_from_simd_t<simd_type> mask)
    {
        using scalar_type = typename simd_type::scalar_t;
        using vector_type = typename simd_type::vector_t;

        if constexpr (fyx::simd::is_128bits_simd_v<simd_type>)
        {
            if constexpr (sizeof(scalar_type) == sizeof(std::uint8_t))
            {
                return simd_type{ _mm_blendv_epi8(source.data, value_to_assign.data, mask.data) };
            }
            else if constexpr (sizeof(scalar_type) == sizeof(std::uint16_t))
            {
                __m128i result = _mm_blendv_epi8(
                    value_to_assign.data, source.data,
                    _mm_cmpeq_epi16(mask.data, _mm_setzero_si128()));
                return simd_type{ result };
            }
            else if constexpr (sizeof(scalar_type) == sizeof(std::uint32_t))
            {
                __m128 result = _mm_blendv_ps(
                    fyx::simd::detail::basic_reinterpret<__m128>(source.data),
                    fyx::simd::detail::basic_reinterpret<__m128>(value_to_assign.data),
                    fyx::simd::detail::basic_reinterpret<__m128>(mask.data)
                );

                return simd_type{ fyx::simd::detail::basic_reinterpret<vector_type>(result) };
            }
            else if constexpr (sizeof(scalar_type) == sizeof(std::uint64_t))
            {
                __m128d result = _mm_blendv_pd(
                    fyx::simd::detail::basic_reinterpret<__m128d>(source.data),
                    fyx::simd::detail::basic_reinterpret<__m128d>(value_to_assign.data),
                    fyx::simd::detail::basic_reinterpret<__m128d>(mask.data)
                );

                return simd_type{ fyx::simd::detail::basic_reinterpret<vector_type>(result) };
            }
            else
            {
                __assume(false);
            }
        }
        else
        {
            if constexpr (sizeof(scalar_type) == sizeof(std::uint8_t))
            {
                return simd_type{ _mm256_blendv_epi8(source.data, value_to_assign.data, mask.data) };
            }
            else if constexpr (sizeof(scalar_type) == sizeof(std::uint16_t))
            {
                __m256i result = _mm256_blendv_epi8(
                    value_to_assign.data, source.data,
                    _mm256_cmpeq_epi16(mask.data, _mm256_setzero_si256()));
                return simd_type{ result };
            }
            else if constexpr (sizeof(scalar_type) == sizeof(std::uint32_t))
            {
                __m256 result = _mm256_blendv_ps(
                    fyx::simd::detail::basic_reinterpret<__m256>(source.data),
                    fyx::simd::detail::basic_reinterpret<__m256>(value_to_assign.data),
                    fyx::simd::detail::basic_reinterpret<__m256>(mask.data)
                );

                return simd_type{ fyx::simd::detail::basic_reinterpret<vector_type>(result) };
            }
            else if constexpr (sizeof(scalar_type) == sizeof(std::uint64_t))
            {
                __m256d result = _mm256_blendv_pd(
                    fyx::simd::detail::basic_reinterpret<__m256d>(source.data),
                    fyx::simd::detail::basic_reinterpret<__m256d>(value_to_assign.data),
                    fyx::simd::detail::basic_reinterpret<__m256d>(mask.data)
                );

                return simd_type{ fyx::simd::detail::basic_reinterpret<vector_type>(result) };
            }
            else
            {
                __assume(false);
            }
        }
    }

    template<typename input_type> requires((fyx::simd::is_basic_simd_v<input_type> || fyx::simd::is_basic_mask_v<input_type>))
    input_type zero_if(input_type source, fyx::simd::mask_from_simd_t<input_type> mask)
    {
        return fyx::simd::where_assign<input_type>(source, fyx::simd::allzero_bits_as<input_type>(), mask);
    }

    template<typename mask_type> requires(fyx::simd::is_basic_mask_v<mask_type>)
    int count_true(mask_type source)
    {
        int mask{ 0 };
        constexpr int interleave_binary = 0b01010101010101010101010101010101;
        if constexpr (fyx::simd::is_128bits_mask_v<mask_type>)
        {
            __m128i value = source.data;
            if constexpr (mask_type::lane_width == 2) { mask = _mm_movemask_pd(_mm_castsi128_pd(value)); }
            else if constexpr (mask_type::lane_width == 4) { mask = _mm_movemask_ps(_mm_castsi128_ps(value)); }
            else if constexpr (mask_type::lane_width == 8)
            {
                mask = _mm_movemask_epi8(value);
                mask &= interleave_binary;
            }
            else if constexpr (mask_type::lane_width == 16)
            {
                mask = _mm_movemask_epi8(value);
            }
            else 
            {
                __assume(false);
            }
        }
        else if constexpr (fyx::simd::is_256bits_mask_v<mask_type>)
        {
            __m256i value = source.data;
            if constexpr (mask_type::lane_width == 4) { mask = _mm256_movemask_pd(_mm256_castsi256_pd(value)); }
            else if constexpr (mask_type::lane_width == 8) { mask = _mm256_movemask_ps(_mm256_castsi256_ps(value)); }
            else if constexpr (mask_type::lane_width == 16)
            {
                mask = _mm256_movemask_epi8(value);
                mask &= interleave_binary;
            }
            else if constexpr (mask_type::lane_width == 32)
            {
                mask = _mm256_movemask_epi8(value);
            }
            else 
            {
                __assume(false);
            }
        }

        return _mm_popcnt_u32(std::bit_cast<unsigned int>(mask));
    }

    template<typename mask_type> requires(fyx::simd::is_basic_mask_v<mask_type>)
    int count_false(mask_type source)
    {
        return mask_type::lane_width - fyx::simd::count_true(source);
    }

    template<typename simd_type> requires(fyx::simd::is_basic_simd_v<simd_type>)
    mask_from_simd_t<simd_type> where_zero(simd_type value)
    {
        const simd_type zero = fyx::simd::allzero_bits_as<simd_type>();
        return fyx::simd::equal(value, zero);
    }

    template<typename mask_type> requires(fyx::simd::is_basic_mask_v<mask_type>)
    bool all_true(mask_type mask)
    {
        if constexpr (fyx::simd::is_128bits_mask_v<mask_type>)
        {
            __m128i value = mask.data;
            return _mm_movemask_epi8(value) == 0xFFFF;
        }
        else if constexpr (fyx::simd::is_256bits_mask_v<mask_type>)
        {
            __m256i value = mask.data;
            return _mm256_movemask_epi8(value) == 0xFFFFFFFF;
        }
        else
        {
            __assume(false);
        }
    }

    template<typename mask_type> requires(fyx::simd::is_basic_mask_v<mask_type>)
    bool any_true(mask_type mask)
    {
        if constexpr (fyx::simd::is_128bits_mask_v<mask_type>)
        {
            __m128i value = mask.data;
            return _mm_movemask_epi8(value) != 0;
        }
        else if constexpr (fyx::simd::is_256bits_mask_v<mask_type>)
        {
            __m256i value = mask.data;
            return _mm256_movemask_epi8(value) != 0;
        }
        else
        {
            __assume(false);
        }
    }

    template<typename mask_type> requires(fyx::simd::is_basic_mask_v<mask_type>)
    bool none_true(mask_type mask)
    {
        return !fyx::simd::any_true(mask);
    }

    template<typename mask_type> requires(fyx::simd::is_basic_mask_v<mask_type>)
    mask_type where_both(mask_type mask_0, mask_type mask_1)
    {
        using uscalar_type = fyx::simd::detail::integral_t<mask_type::single_width_bits, false>;
        using usimd_type = basic_simd<uscalar_type, mask_type::bit_width>;
        return mask_type{ fyx::simd::bitwise_AND(usimd_type{ mask_0.data }, usimd_type{ mask_1.data }) };
    }

    template<typename mask_type> requires(fyx::simd::is_basic_mask_v<mask_type>)
    mask_type where_any(mask_type mask_0, mask_type mask_1)
    {
        using uscalar_type = fyx::simd::detail::integral_t<mask_type::single_width_bits, false>;
        using usimd_type = basic_simd<uscalar_type, mask_type::bit_width>;
        return mask_type{ fyx::simd::bitwise_OR(usimd_type{ mask_0.data }, usimd_type{ mask_1.data }) };
    }

    template<typename mask_type> requires(fyx::simd::is_basic_mask_v<mask_type>)
    mask_type where_either(mask_type mask_0, mask_type mask_1)
    {
        using uscalar_type = fyx::simd::detail::integral_t<mask_type::single_width_bits, false>;
        using usimd_type = basic_simd<uscalar_type, mask_type::bit_width>;
        return mask_type{ fyx::simd::bitwise_XOR(usimd_type{ mask_0.data }, usimd_type{ mask_1.data }) };
    }

    template<typename mask_type> requires(fyx::simd::is_basic_mask_v<mask_type>)
    mask_type where_invert(mask_type target_mask, mask_type condition_mask)
    {
        using uscalar_type = fyx::simd::detail::integral_t<mask_type::single_width_bits, false>;
        using usimd_type = basic_simd<uscalar_type, mask_type::bit_width>;
        return mask_type{ fyx::simd::bitwise_XOR(usimd_type{ target_mask.data }, usimd_type{ condition_mask.data }) };
    }

    template<typename simd_type, typename mask_type>
        requires(fyx::simd::is_basic_simd_v<simd_type>&& fyx::simd::is_basic_mask_v<mask_type>
#if !defined(_FOYE_SIMD_ENABLE_EMULATED_)
    && (simd_type::scalar_bit_width == 32 || simd_type::scalar_bit_width == 64)
#endif
        && ((simd_type::lane_width == mask_type::lane_width)
            && simd_type::bit_width == mask_type::bit_width))
    void where_store(simd_type to_store_vector, mask_type mask, void* mem_addr)
    {
        if constexpr (simd_type::scalar_bit_width == 32)
        {
            if constexpr (is_256bits_simd_v<simd_type>)
            {
                _mm256_maskstore_epi32(reinterpret_cast<int*>(mem_addr),
                    detail::basic_reinterpret<__m256i>(mask.data),
                    detail::basic_reinterpret<__m256i>(to_store_vector.data));
            }
            else
            {
                _mm_maskstore_epi32(reinterpret_cast<int*>(mem_addr),
                    detail::basic_reinterpret<__m128i>(mask.data),
                    detail::basic_reinterpret<__m128i>(to_store_vector.data));
            }
        }
        else if constexpr (simd_type::scalar_bit_width == 64)
        {
            if constexpr (is_256bits_simd_v<simd_type>)
            {
                _mm256_maskstore_epi64(reinterpret_cast<long long*>(mem_addr),
                    detail::basic_reinterpret<__m256i>(mask.data),
                    detail::basic_reinterpret<__m256i>(to_store_vector.data));
            }
            else
            {
                _mm_maskstore_epi64(reinterpret_cast<long long*>(mem_addr),
                    detail::basic_reinterpret<__m128i>(mask.data),
                    detail::basic_reinterpret<__m128i>(to_store_vector.data));
            }
        }
        else
        {
#if defined(_FOYE_SIMD_ENABLE_EMULATED_)
#if !defined(_FOYE_SIMD_DISABLE_MASK_LOAD_ERROR)
            static_assert(simd_type::scalar_bit_width == 32 || simd_type::scalar_bit_width == 64,
                "Implemented by a large combination of composite instructions, "
                "there are serious performance issues compared to native instructions"
                "define _FOYE_SIMD_DISABLE_MASK_LOAD_ERROR to close this error message and enable simulation implementation");
#else
            simd_type source = load_unaligned<simd_type>(
                reinterpret_cast<const typename simd_type::scalar_t*>(mem_addr));
            simd_type result = where_assign(source, to_store_vector, mask);
            store_unaligned(result, mem_addr);
#endif
#else
            __assume(false);
#endif
        }
    }

    template<typename simd_type, int index_scale = sizeof(typename simd_type::scalar_t)>
    requires((simd_type::scalar_bit_width == 32) && is_256bits_simd_v<simd_type>)
    simd_type load_gather_32i32o(sint32x8 indices, void* mem_addr)
    {
        __m256 float_result = _mm256_i32gather_ps(reinterpret_cast<const float*>(mem_addr),
            indices.data, index_scale);
        return simd_type{ detail::basic_reinterpret<typename simd_type::vector_t>(float_result) };
    }

    namespace detail
    {
        static const std::array<std::array<std::uint8_t, 16>, 65536>& bytewise_forward_lut = []() -> auto& 
            {
                static auto* lut = new std::array<std::array<std::uint8_t, 16>, 65536>;

                for (int bitmask = 0; bitmask < 65536; bitmask++)
                {
                    std::array<std::uint8_t, 16> control{};
                    int j = 0;

                    for (int i = 0; i < 16; i++)
                    {
                        if (bitmask & (1 << i))
                        {
                            control[j++] = static_cast<std::uint8_t>(i);
                        }
                    }

                    for (; j < 16; j++)
                    {
                        control[j] = 0;
                    }

                    (*lut)[bitmask] = control;
                }

                return *lut;
            }();
    }

    template<typename simd_type>
    requires(is_128bits_simd_v<simd_type>&& simd_type::scalar_bit_width == 8)
    simd_type bytewise_forward(simd_type source, mask_8x16 select_mask)
    {
        __m128i mask = _mm_cmpeq_epi8(select_mask.data, _mm_setzero_si128());
        int bitmask = _mm_movemask_epi8(_mm_xor_si128(mask, _mm_set1_epi8(0xFF)));

        const std::uint8_t* lut_addr = detail::bytewise_forward_lut[bitmask].data();
        __m128i shuffle_mask = _mm_load_si128(reinterpret_cast<const __m128i*>(lut_addr));
        return simd_type{ _mm_shuffle_epi8(source.data, shuffle_mask) };
    }



    uint8x16 reverse(uint8x16 input)
    {
        const __m128i mask = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        return uint8x16{ _mm_shuffle_epi8(input.data, mask) };
    }

    uint16x8 reverse(uint16x8 input) 
    {
        const __m128i mask = _mm_set_epi8(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14);
        return uint16x8{ _mm_shuffle_epi8(input.data, mask) }; }
    
    uint32x4 reverse(uint32x4 input) 
    {
        constexpr int _imm = _MM_SHUFFLE(0, 1, 2, 3);
        return uint32x4(_mm_shuffle_epi32(input.data, _imm));
    }
    
    uint64x2 reverse(uint64x2 input)
    {
        constexpr int _imm = _MM_SHUFFLE(1, 0, 3, 2);
        return uint64x2{ _mm_shuffle_epi32(input.data, _imm) };
    }

    sint8x16 reverse(sint8x16 input) { return sint8x16{ fyx::simd::reverse(uint8x16{input.data}) }; }
    sint16x8 reverse(sint16x8 input) { return sint16x8{ fyx::simd::reverse(uint16x8{input.data}) }; }
    sint32x4 reverse(sint32x4 input) { return sint32x4{ fyx::simd::reverse(uint32x4{input.data}) }; }
    sint64x2 reverse(sint64x2 input) { return sint64x2{ fyx::simd::reverse(uint64x2{input.data}) }; }

    float32x4 reverse(float32x4 input)
    {
        constexpr int _imm = _MM_SHUFFLE(0, 1, 2, 3);
        return float32x4{ _mm_shuffle_ps(input.data, input.data, _imm) };
    }

    float64x2 reverse(float64x2 input)
    {
        return float64x2{ _mm_shuffle_pd(input.data, input.data, 0b01) };
    }

    uint8x32 reverse(uint8x32 input)
    {
        const __m256i mask = _mm256_set_epi8(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        return uint8x32{ _mm256_shuffle_epi8(
            _mm256_permute2x128_si256(input.data, input.data, 0x01), mask) };
    }

    uint16x16 reverse(uint16x16 input)
    {
        const __m256i mask = _mm256_set_epi8(
            1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14,
            1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14);
        return uint16x16{ _mm256_shuffle_epi8(
            _mm256_permute2x128_si256(input.data, input.data, 0x01), mask) };
    }

    uint32x8 reverse(uint32x8 input)
    {
        constexpr int _imm = _MM_SHUFFLE(0, 1, 2, 3);
        return uint32x8{ _mm256_shuffle_epi32(
            _mm256_permute2x128_si256(input.data, input.data, 0x01), _imm) };
    }

    uint64x4 reverse(uint64x4 input)
    {
        constexpr int _imm = _MM_SHUFFLE(0, 1, 2, 3);
        return uint64x4{ _mm256_permute4x64_epi64(input.data, _imm) };
    }

    sint8x32 reverse(sint8x32 input) { return sint8x32{ fyx::simd::reverse(uint8x32{input.data}) }; }
    sint16x16 reverse(sint16x16 input) { return sint16x16{ fyx::simd::reverse(uint16x16{input.data}) }; }
    sint32x8 reverse(sint32x8 input) { return sint32x8{ fyx::simd::reverse(uint32x8{input.data}) }; }
    sint64x4 reverse(sint64x4 input) { return sint64x4{ fyx::simd::reverse(uint64x4{input.data}) }; }

    float32x8 reverse(float32x8 input)
    {
        constexpr int _imm = _MM_SHUFFLE(0, 1, 2, 3);
        __m256 swapped = _mm256_permute2f128_ps(input.data, input.data, 0x01);
        return float32x8{ _mm256_permute_ps(swapped, _imm) };
    }

    float64x4 reverse(float64x4 input)
    {
        constexpr int _imm = _MM_SHUFFLE(0, 1, 2, 3);
        return float64x4{ _mm256_permute4x64_pd(input.data, _imm) };
    }

#if defined(_FOYE_SIMD_HAS_FP16_)
    float16x8 reverse(float16x8 input) { return float16x8{ fyx::simd::reverse(uint16x8{input.data}) }; }
    float16x16 reverse(float16x16 input) { return float16x16{ fyx::simd::reverse(uint16x16{input.data}) }; }
#endif
#if defined(_FOYE_SIMD_HAS_FP16_)
    bfloat16x8 reverse(bfloat16x8 input) { return bfloat16x8{ fyx::simd::reverse(uint16x8{input.data}) }; }
    bfloat16x16 reverse(bfloat16x16 input) { return bfloat16x16{ fyx::simd::reverse(uint16x16{input.data}) }; }
#endif

    uint8x16 swap_halves(uint8x16 input) { return uint8x16{ _mm_or_si128(_mm_slli_si128(input.data, 8), _mm_srli_si128(input.data, 8)) }; }
    uint16x8 swap_halves(uint16x8 input) { return uint16x8{ _mm_or_si128(_mm_slli_si128(input.data, 8), _mm_srli_si128(input.data, 8)) }; }
    uint32x4 swap_halves(uint32x4 input) { return uint32x4{ _mm_or_si128(_mm_slli_si128(input.data, 8), _mm_srli_si128(input.data, 8)) }; }
    uint64x2 swap_halves(uint64x2 input) { return uint64x2{ _mm_or_si128(_mm_slli_si128(input.data, 8), _mm_srli_si128(input.data, 8)) }; }
    sint8x16 swap_halves(sint8x16 input) { return sint8x16{ _mm_or_si128(_mm_slli_si128(input.data, 8), _mm_srli_si128(input.data, 8)) }; }
    sint16x8 swap_halves(sint16x8 input) { return sint16x8{ _mm_or_si128(_mm_slli_si128(input.data, 8), _mm_srli_si128(input.data, 8)) }; }
    sint32x4 swap_halves(sint32x4 input) { return sint32x4{ _mm_or_si128(_mm_slli_si128(input.data, 8), _mm_srli_si128(input.data, 8)) }; }
    sint64x2 swap_halves(sint64x2 input) { return sint64x2{ _mm_or_si128(_mm_slli_si128(input.data, 8), _mm_srli_si128(input.data, 8)) }; }
    float32x4 swap_halves(float32x4 input) { return float32x4{ _mm_shuffle_ps(input.data, input.data, _MM_SHUFFLE(1, 0, 3, 2)) }; }
    float64x2 swap_halves(float64x2 input) { return float64x2{ _mm_shuffle_pd(input.data, input.data, 0b01) }; }
    uint8x32 swap_halves(uint8x32 input) { return uint8x32{ detail::merge(detail::split_high(input.data), detail::split_low(input.data)) }; }
    uint16x16 swap_halves(uint16x16 input) { return uint16x16{ detail::merge(detail::split_high(input.data), detail::split_low(input.data)) }; }
    uint32x8 swap_halves(uint32x8 input) { return uint32x8{ detail::merge(detail::split_high(input.data), detail::split_low(input.data)) }; }
    uint64x4 swap_halves(uint64x4 input) { return uint64x4{ detail::merge(detail::split_high(input.data), detail::split_low(input.data)) }; }
    sint8x32 swap_halves(sint8x32 input) { return sint8x32{ detail::merge(detail::split_high(input.data), detail::split_low(input.data)) }; }
    sint16x16 swap_halves(sint16x16 input) { return sint16x16{ detail::merge(detail::split_high(input.data), detail::split_low(input.data)) }; }
    sint32x8 swap_halves(sint32x8 input) { return sint32x8{ detail::merge(detail::split_high(input.data), detail::split_low(input.data)) }; }
    sint64x4 swap_halves(sint64x4 input) { return sint64x4{ detail::merge(detail::split_high(input.data), detail::split_low(input.data)) }; }
    float32x8 swap_halves(float32x8 input) { return float32x8{ detail::merge(detail::split_high(input.data), detail::split_low(input.data)) }; }
    float64x4 swap_halves(float64x4 input) { return float64x4{ detail::merge(detail::split_high(input.data), detail::split_low(input.data)) }; }
#if defined(_FOYE_SIMD_HAS_FP16_)
    float16x8 swap_halves(float16x8 input) { return float16x8{ fyx::simd::swap_halves(uint16x8{input.data}) }; }
    float16x16 swap_halves(float16x16 input) { return float16x16{ fyx::simd::swap_halves(uint16x16{input.data}) }; }
#endif
#if defined(_FOYE_SIMD_HAS_FP16_)
    bfloat16x8 swap_halves(bfloat16x8 input) { return bfloat16x8{ fyx::simd::swap_halves(uint16x8{input.data}) }; }
    bfloat16x16 swap_halves(bfloat16x16 input) { return bfloat16x16{ fyx::simd::swap_halves(uint16x16{input.data}) }; }
#endif
}

namespace fyx::simd
{
    template<typename mask_type> requires(is_basic_mask_v<mask_type>)
    mask_type operator & (mask_type lhs, mask_type rhs)
    {
        return where_both(lhs, rhs);
    }

    template<typename mask_type> requires(is_basic_mask_v<mask_type>)
    mask_type operator | (mask_type lhs, mask_type rhs)
    {
        return where_any(lhs, rhs);
    }

    template<typename mask_type> requires(is_basic_mask_v<mask_type>)
    mask_type operator ^ (mask_type lhs, mask_type rhs)
    {
        return where_either(lhs, rhs);
    }

    template<typename mask_type> requires(is_basic_mask_v<mask_type>)
    mask_type operator == (mask_type lhs, mask_type rhs)
    {
        using temp_simd_t = basic_simd<detail::integral_t<
            mask_type::single_width_bits, false>, mask_8x32::bit_width>;
        return equal(lhs.as_basic_simd<temp_simd_t>(), rhs.as_basic_simd<temp_simd_t>());
    }

    template<typename mask_type> requires(is_basic_mask_v<mask_type>)
    mask_type operator != (mask_type lhs, mask_type rhs)
    {
        using temp_simd_t = basic_simd<detail::integral_t<
            mask_type::single_width_bits, false>, mask_8x32::bit_width>;
        return not_equal(lhs.as_basic_simd<temp_simd_t>(), rhs.as_basic_simd<temp_simd_t>());
    }

    template<typename mask_type> requires(is_basic_mask_v<mask_type>)
    mask_type& operator &= (mask_type& lhs, mask_type rhs)
    {
        lhs = where_both(lhs, rhs);
        return lhs;
    }

    template<typename mask_type> requires(is_basic_mask_v<mask_type>)
    mask_type& operator |= (mask_type& lhs, mask_type rhs)
    {
        lhs = where_any(lhs, rhs);
        return lhs;
    }

    template<typename mask_type> requires(is_basic_mask_v<mask_type>)
    mask_type& operator ^= (mask_type& lhs, mask_type rhs)
    {
        lhs = where_either(lhs, rhs);
        return lhs;
    }

    template<typename mask_type> requires(is_basic_mask_v<mask_type>)
    mask_type operator ! (mask_type mask)
    {
        return where_invert(allone_bits_as<mask_type>(), mask);
    }
}

#endif
