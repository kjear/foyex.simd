#ifndef _FOYE_SIMD_SHIFT_HPP_
#define _FOYE_SIMD_SHIFT_HPP_
#pragma once

#include "simd_def.hpp"
#include "simd_utility.hpp"
#include "simd_cvt.hpp"

namespace fyx::simd // left shift by runtime same shift amount, logical for both unsigned and signed
{
    uint16x8 shift_left(uint16x8 input, std::uint16_t shift_amount)
    {
        __m128i count = _mm_cvtsi32_si128(static_cast<int>(shift_amount));
        __m128i result = _mm_sll_epi16(input.data, count);
        return uint16x8{ result };
    }

    uint32x4 shift_left(uint32x4 input, std::uint32_t shift_amount)
    {
        __m128i count = _mm_cvtsi32_si128(std::bit_cast<int>(shift_amount));
        __m128i result = _mm_sll_epi32(input.data, count);
        return uint32x4{ result };
    }

    uint64x2 shift_left(uint64x2 input, std::uint64_t shift_amount)
    {
        __m128i count = _mm_set1_epi64x(std::bit_cast<long long>(shift_amount));
        __m128i result = _mm_sll_epi64(input.data, count);
        return uint64x2{ result };
    }

    uint16x16 shift_left(uint16x16 input, std::uint16_t shift_amount)
    {
        __m128i count = _mm_cvtsi32_si128(static_cast<int>(shift_amount));
        __m256i result = _mm256_sll_epi16(input.data, count);
        return uint16x16{ result };
    }

    uint32x8 shift_left(uint32x8 input, std::uint32_t shift_amount)
    {
        __m128i count = _mm_cvtsi32_si128(std::bit_cast<int>(shift_amount));
        __m256i result = _mm256_sll_epi32(input.data, count);
        return uint32x8{ result };
    }

    uint64x4 shift_left(uint64x4 input, std::uint64_t shift_amount)
    {
        __m128i count = _mm_set1_epi64x(std::bit_cast<long long>(shift_amount));
        __m256i result = _mm256_sll_epi64(input.data, count);
        return uint64x4{ result };
    }

    sint16x8 shift_left(sint16x8 input, std::uint16_t shift_amount)
    {
        return fyx::simd::reinterpret<sint16x8>(
            fyx::simd::shift_left(fyx::simd::reinterpret<as_unsigned_type<sint16x8>>(input), 
                shift_amount));
    }

    sint32x4 shift_left(sint32x4 input, std::uint16_t shift_amount)
    {
        return fyx::simd::reinterpret<sint32x4>(
            fyx::simd::shift_left(fyx::simd::reinterpret<as_unsigned_type<sint32x4>>(input), 
                shift_amount));
    }

    sint64x2 shift_left(sint64x2 input, std::uint16_t shift_amount)
    {
        return fyx::simd::reinterpret<sint64x2>(
            fyx::simd::shift_left(fyx::simd::reinterpret<as_unsigned_type<sint64x2>>(input), 
                shift_amount));
    }

    sint16x16 shift_left(sint16x16 input, std::uint16_t shift_amount)
    {
        return fyx::simd::reinterpret<sint16x16>(
            fyx::simd::shift_left(fyx::simd::reinterpret<as_unsigned_type<sint16x16>>(input), 
                shift_amount));
    }

    sint32x8 shift_left(sint32x8 input, std::uint16_t shift_amount)
    {
        return fyx::simd::reinterpret<sint32x8>(
            fyx::simd::shift_left(fyx::simd::reinterpret<as_unsigned_type<sint32x8>>(input), 
                shift_amount));
    }

    sint64x4 shift_left(sint64x4 input, std::uint16_t shift_amount)
    {
        return fyx::simd::reinterpret<sint64x4>(
            fyx::simd::shift_left(fyx::simd::reinterpret<as_unsigned_type<sint64x4>>(input), 
                shift_amount));
    }

#if defined(_FOYE_SIMD_ENABLE_EMULATED_)
    uint8x16 shift_left(uint8x16 input, std::uint8_t shift_amount)
    {
        uint16x16 input16x16 = fyx::simd::expand<uint16x16>(input);
        uint16x16 result = fyx::simd::shift_left(input16x16, static_cast<std::uint16_t>(shift_amount));
        return fyx::simd::narrowing<uint8x16>(result);
    }

    uint8x32 shift_left(uint8x32 input, std::uint8_t shift_amount)
    {
        const std::uint16_t amount = static_cast<std::uint16_t>(shift_amount);
        uint16x16 result_low = fyx::simd::shift_left(fyx::simd::expand_low<uint16x16>(input), amount);
        uint16x16 result_high = fyx::simd::shift_left(fyx::simd::expand_high<uint16x16>(input), amount);

        return fyx::simd::merge(
            fyx::simd::narrowing<uint8x16>(result_low),
            fyx::simd::narrowing<uint8x16>(result_high)
        );
    }

    sint8x16 shift_left(sint8x16 input, std::uint8_t shift_amount)
    {
        return fyx::simd::reinterpret<sint8x16>(
            fyx::simd::shift_left(fyx::simd::reinterpret<as_unsigned_type<sint8x16>>(input), 
                shift_amount));
    }

    sint8x32 shift_left(sint8x32 input, std::uint8_t shift_amount)
    {
        return fyx::simd::reinterpret<sint8x32>(
            fyx::simd::shift_left(fyx::simd::reinterpret<as_unsigned_type<sint8x32>>(input), 
                shift_amount));
    }
#endif
}

namespace fyx::simd // right shift by runtime same shift amount, arithmetic for signed and logical for unsigned
{
    uint16x8 shift_right(uint16x8 input, std::uint16_t shift_amount)
    {
        __m128i count = _mm_cvtsi32_si128(static_cast<int>(shift_amount));
        __m128i result = _mm_srl_epi16(input.data, count);
        return uint16x8{ result };
    }

    uint32x4 shift_right(uint32x4 input, std::uint32_t shift_amount)
    {
        __m128i count = _mm_cvtsi32_si128(std::bit_cast<int>(shift_amount));
        __m128i result = _mm_srl_epi32(input.data, count);
        return uint32x4{ result };
    }

    uint64x2 shift_right(uint64x2 input, std::uint64_t shift_amount)
    {
        __m128i count = _mm_set1_epi64x(std::bit_cast<long long>(shift_amount));
        __m128i result = _mm_srl_epi64(input.data, count);
        return uint64x2{ result };
    }

    uint16x16 shift_right(uint16x16 input, std::uint16_t shift_amount)
    {
        __m128i count = _mm_cvtsi32_si128(static_cast<int>(shift_amount));
        __m256i result = _mm256_srl_epi16(input.data, count);
        return uint16x16{ result };
    }

    uint32x8 shift_right(uint32x8 input, std::uint32_t shift_amount)
    {
        __m128i count = _mm_cvtsi32_si128(std::bit_cast<int>(shift_amount));
        __m256i result = _mm256_srl_epi32(input.data, count);
        return uint32x8{ result };
    }

    uint64x4 shift_right(uint64x4 input, std::uint64_t shift_amount)
    {
        __m128i count = _mm_set1_epi64x(std::bit_cast<long long>(shift_amount));
        __m256i result = _mm256_srl_epi64(input.data, count);
        return uint64x4{ result };
    }
#if defined(_FOYE_SIMD_ENABLE_EMULATED_)
    uint8x16 shift_right(uint8x16 input, std::uint8_t shift_amount)
    {
        uint16x16 input16x16 = fyx::simd::expand<uint16x16>(input);
        uint16x16 result = fyx::simd::shift_right(input16x16, static_cast<std::uint16_t>(shift_amount));
        return fyx::simd::narrowing<uint8x16>(result);
    }

    uint8x32 shift_right(uint8x32 input, std::uint8_t shift_amount)
    {
        const std::uint16_t amount = static_cast<std::uint16_t>(shift_amount);
        uint16x16 result_low = fyx::simd::shift_right(fyx::simd::expand_low<uint16x16>(input), amount);
        uint16x16 result_high = fyx::simd::shift_right(fyx::simd::expand_high<uint16x16>(input), amount);

        return fyx::simd::merge(
            fyx::simd::narrowing<uint8x16>(result_low),
            fyx::simd::narrowing<uint8x16>(result_high)
        );
    }
#endif

    sint16x8 shift_right(sint16x8 input, std::uint16_t shift_amount)
    {
        __m128i count = _mm_cvtsi32_si128(static_cast<int>(shift_amount));
        __m128i result = _mm_sra_epi16(input.data, count);
        return sint16x8{ result };
    }

    sint32x4 shift_right(sint32x4 input, std::uint32_t shift_amount)
    {
        __m128i count = _mm_cvtsi32_si128(std::bit_cast<int>(shift_amount));
        __m128i result = _mm_sra_epi32(input.data, count);
        return sint32x4{ result };
    }

    sint64x2 shift_right(sint64x2 input, std::uint64_t shift_amount)
    {
        alignas(32) sint64x2::scalar_t arr[2];
        store_aligned(input, arr);
        return sint64x2{ _mm_setr_epi64x(arr[0] >> shift_amount, arr[1] >> shift_amount) };
    }


    sint16x16 shift_right(sint16x16 input, std::uint16_t shift_amount)
    {
        __m128i count = _mm_cvtsi32_si128(static_cast<int>(shift_amount));
        __m256i result = _mm256_sra_epi16(input.data, count);
        return sint16x16{ result };
    }

    sint32x8 shift_right(sint32x8 input, std::uint32_t shift_amount)
    {
        __m128i count = _mm_cvtsi32_si128(std::bit_cast<int>(shift_amount));
        __m256i result = _mm256_sra_epi32(input.data, count);
        return sint32x8{ result };
    }

    sint64x4 shift_right(sint64x4 input, std::uint64_t shift_amount)
    {
        alignas(32) sint64x4::scalar_t arr[4];
        store_aligned(input, arr);
        return sint64x4{ _mm256_setr_epi64x(
            arr[0] >> shift_amount, arr[1] >> shift_amount,
            arr[2] >> shift_amount, arr[3] >> shift_amount
        ) };
    }
#if defined(_FOYE_SIMD_ENABLE_EMULATED_)
    sint8x16 shift_right(sint8x16 input, std::uint8_t shift_amount)
    {
        sint16x16 input16x16 = fyx::simd::expand<sint16x16>(input);
        sint16x16 result = fyx::simd::shift_right(input16x16, static_cast<std::uint16_t>(shift_amount));
        return fyx::simd::narrowing<sint8x16>(result);
    }
    sint8x32 shift_right(sint8x32 input, std::uint8_t shift_amount)
    {
        const std::uint16_t amount = static_cast<std::uint16_t>(shift_amount);
        sint16x16 result_low = fyx::simd::shift_right(fyx::simd::expand_low<sint16x16>(input), amount);
        sint16x16 result_high = fyx::simd::shift_right(fyx::simd::expand_high<sint16x16>(input), amount);

        return fyx::simd::merge(
            fyx::simd::narrowing<sint8x16>(result_low),
            fyx::simd::narrowing<sint8x16>(result_high)
        );
    }
#endif
}

namespace fyx::simd // left shift by compile-time same shift amount, logical for both signed and unsigned
{
#if defined(_FOYE_SIMD_ENABLE_EMULATED_)
    template<int shift_amount> requires(shift_amount >= 0 && shift_amount <= 7)
        uint8x16 shift_left(uint8x16 input)
    {
        const __m128i zerobits = _mm_set1_epi16(0x00FF);
        __m128i low16 = _mm_slli_epi16(_mm_cvtepu8_epi16(input.data), shift_amount);
        __m128i high16 = _mm_slli_epi16(_mm_cvtepu8_epi16(_mm_srli_si128(input.data, 8)), shift_amount);

        return uint8x16{ _mm_packus_epi16(
            _mm_and_si128(low16, zerobits),
            _mm_and_si128(high16, zerobits)) };
    }
#endif
    template<int shift_amount> requires(shift_amount >= 0 && shift_amount <= 15)
        uint16x8 shift_left(uint16x8 input) { return uint16x8{ _mm_slli_epi16(input.data, shift_amount) }; }

    template<int shift_amount> requires(shift_amount >= 0 && shift_amount <= 31)
        uint32x4 shift_left(uint32x4 input) { return uint32x4{ _mm_slli_epi32(input.data, shift_amount) }; }

    template<int shift_amount> requires(shift_amount >= 0 && shift_amount <= 63)
        uint64x2 shift_left(uint64x2 input) { return uint64x2{ _mm_slli_epi64(input.data, shift_amount) }; }

#if defined(_FOYE_SIMD_ENABLE_EMULATED_)
    template<int shift_amount> requires(shift_amount >= 0 && shift_amount <= 7)
        uint8x32 shift_left(uint8x32 input)
    {
        const __m256i mask = _mm256_set1_epi16(0x00FF);

        __m256i low16 = _mm256_slli_epi16(_mm256_cvtepu8_epi16(_mm256_castsi256_si128(input.data)), shift_amount);
        __m256i high16 = _mm256_slli_epi16(_mm256_cvtepu8_epi16(_mm256_extracti128_si256(input.data, 1)), shift_amount);

        return uint8x32{ _mm256_permute4x64_epi64(
            _mm256_packus_epi16(
                _mm256_and_si256(low16, mask),
                _mm256_and_si256(high16, mask)),
            0b11011000) };
    }
#endif
    template<int shift_amount> requires(shift_amount >= 0 && shift_amount <= 15)
        uint16x16 shift_left(uint16x16 input) { return uint16x16{ _mm256_slli_epi16(input.data, shift_amount) }; }

    template<int shift_amount> requires(shift_amount >= 0 && shift_amount <= 31)
        uint32x8 shift_left(uint32x8 input) { return uint32x8{ _mm256_slli_epi32(input.data, shift_amount) }; }

    template<int shift_amount> requires(shift_amount >= 0 && shift_amount <= 63)
        uint64x4 shift_left(uint64x4 input) { return uint64x4{ _mm256_slli_epi64(input.data, shift_amount) }; }


#if defined(_FOYE_SIMD_ENABLE_EMULATED_)
    template<int shift_amount> requires(shift_amount >= 0 && shift_amount <= 7)
        sint8x16 shift_left(sint8x16 input)
    {
        const __m128i zerobits = _mm_set1_epi16(0x00FF);
        __m128i low16 = _mm_slli_epi16(_mm_cvtepi8_epi16(input.data), shift_amount);
        __m128i high16 = _mm_slli_epi16(_mm_cvtepi8_epi16(_mm_srli_si128(input.data, 8)), shift_amount);

        return sint8x16{ _mm_packus_epi16(
            _mm_and_si128(low16, zerobits),
            _mm_and_si128(high16, zerobits)) };
    }
#endif
    template<int shift_amount> requires(shift_amount >= 0 && shift_amount <= 15)
        sint16x8 shift_left(sint16x8 input) { return sint16x8{ _mm_slli_epi16(input.data, shift_amount) }; }

    template<int shift_amount> requires(shift_amount >= 0 && shift_amount <= 31)
        sint32x4 shift_left(sint32x4 input) { return sint32x4{ _mm_slli_epi32(input.data, shift_amount) }; }

    template<int shift_amount> requires(shift_amount >= 0 && shift_amount <= 63)
        sint64x2 shift_left(sint64x2 input) { return sint64x2{ _mm_slli_epi64(input.data, shift_amount) }; }

#if defined(_FOYE_SIMD_ENABLE_EMULATED_)
    template<int shift_amount> requires(shift_amount >= 0 && shift_amount <= 7)
        sint8x32 shift_left(sint8x32 input)
    {
        const __m256i mask = _mm256_set1_epi16(0x00FF);
        __m256i low16 = _mm256_slli_epi16(_mm256_cvtepi8_epi16(_mm256_castsi256_si128(input.data)), shift_amount);
        __m256i high16 = _mm256_slli_epi16(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(input.data, 1)), shift_amount);

        return sint8x32{ _mm256_permute4x64_epi64(
            _mm256_packus_epi16(
                _mm256_and_si256(low16, mask),
                _mm256_and_si256(high16, mask)),
            0b11011000) };
    }
#endif
    template<int shift_amount> requires(shift_amount >= 0 && shift_amount <= 15)
        sint16x16 shift_left(sint16x16 input) { return sint16x16{ _mm256_slli_epi16(input.data, shift_amount) }; }

    template<int shift_amount> requires(shift_amount >= 0 && shift_amount <= 31)
        sint32x8 shift_left(sint32x8 input) { return sint32x8{ _mm256_slli_epi32(input.data, shift_amount) }; }

    template<int shift_amount> requires(shift_amount >= 0 && shift_amount <= 63)
        sint64x4 shift_left(sint64x4 input) { return sint64x4{ _mm256_slli_epi64(input.data, shift_amount) }; }
}

namespace fyx::simd // right shift by compile-time same shift amount, arithmetic for signed and logical for unsigned
{
#if defined(_FOYE_SIMD_ENABLE_EMULATED_)
    template<int shift_amount> requires(shift_amount >= 0 && shift_amount <= 7)
        uint8x16 shift_right(uint8x16 input)
    {
        const __m128i zerobits = _mm_set1_epi16(0x00FF);
        __m128i low16 = _mm_srli_epi16(_mm_cvtepu8_epi16(input.data), shift_amount);
        __m128i high16 = _mm_srli_epi16(_mm_cvtepu8_epi16(_mm_srli_si128(input.data, 8)), shift_amount);

        return uint8x16{ _mm_packus_epi16(
            _mm_and_si128(low16, zerobits),
            _mm_and_si128(high16, zerobits)) };
    }
#endif
    template<int shift_amount> requires(shift_amount >= 0 && shift_amount <= 15)
        uint16x8 shift_right(uint16x8 input) { return uint16x8{ _mm_srli_epi16(input.data, shift_amount) }; }

    template<int shift_amount> requires(shift_amount >= 0 && shift_amount <= 31)
        uint32x4 shift_right(uint32x4 input) { return uint32x4{ _mm_srli_epi32(input.data, shift_amount) }; }

    template<int shift_amount> requires(shift_amount >= 0 && shift_amount <= 63)
        uint64x2 shift_right(uint64x2 input) { return uint64x2{ _mm_srli_epi64(input.data, shift_amount) }; }

#if defined(_FOYE_SIMD_ENABLE_EMULATED_)
    template<int shift_amount> requires(shift_amount >= 0 && shift_amount <= 7)
        uint8x32 shift_right(uint8x32 input)
    {
        const __m256i mask = _mm256_set1_epi16(0x00FF);

        __m256i low16 = _mm256_srli_epi16(_mm256_cvtepu8_epi16(_mm256_castsi256_si128(input.data)), shift_amount);
        __m256i high16 = _mm256_srli_epi16(_mm256_cvtepu8_epi16(_mm256_extracti128_si256(input.data, 1)), shift_amount);

        return uint8x32{ _mm256_permute4x64_epi64(
            _mm256_packus_epi16(
                _mm256_and_si256(low16, mask),
                _mm256_and_si256(high16, mask)),
            0b11011000) };
    }
#endif
    template<int shift_amount> requires(shift_amount >= 0 && shift_amount <= 15)
        uint16x16 shift_right(uint16x16 input) { return uint16x16{ _mm256_srli_epi16(input.data, shift_amount) }; }

    template<int shift_amount> requires(shift_amount >= 0 && shift_amount <= 31)
        uint32x8 shift_right(uint32x8 input) { return uint32x8{ _mm256_srli_epi32(input.data, shift_amount) }; }

    template<int shift_amount> requires(shift_amount >= 0 && shift_amount <= 63)
        uint64x4 shift_right(uint64x4 input) { return uint64x4{ _mm256_srli_epi64(input.data, shift_amount) }; }


#if defined(_FOYE_SIMD_ENABLE_EMULATED_)
    template<int shift_amount> requires(shift_amount >= 0 && shift_amount <= 7)
        sint8x16 shift_right(sint8x16 input)
    {
        const __m128i zerobits = _mm_set1_epi16(0x00FF);
        __m128i low16 = _mm_srai_epi16(_mm_cvtepi8_epi16(input.data), shift_amount);
        __m128i high16 = _mm_srai_epi16(_mm_cvtepi8_epi16(_mm_srli_si128(input.data, 8)), shift_amount);

        return sint8x16{ _mm_packus_epi16(
            _mm_and_si128(low16, zerobits),
            _mm_and_si128(high16, zerobits)) };
    }
#endif
    template<int shift_amount> requires(shift_amount >= 0 && shift_amount <= 15)
        sint16x8 shift_right(sint16x8 input) { return sint16x8{ _mm_srai_epi16(input.data, shift_amount) }; }

    template<int shift_amount> requires(shift_amount >= 0 && shift_amount <= 31)
        sint32x4 shift_right(sint32x4 input) { return sint32x4{ _mm_srai_epi32(input.data, shift_amount) }; }

    template<int shift_amount> requires(shift_amount >= 0 && shift_amount <= 63)
        sint64x2 shift_right(sint64x2 input) { return sint64x2{ _mm_srai_epi64(input.data, shift_amount) }; }

#if defined(_FOYE_SIMD_ENABLE_EMULATED_)
    template<int shift_amount> requires(shift_amount >= 0 && shift_amount <= 7)
        sint8x32 shift_right(sint8x32 input)
    {
        const __m256i mask = _mm256_set1_epi16(0x00FF);

        __m256i low16 = _mm256_srai_epi16(_mm256_cvtepi8_epi16(_mm256_castsi256_si128(input.data)), shift_amount);
        __m256i high16 = _mm256_srai_epi16(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(input.data, 1)), shift_amount);

        return sint8x32{ _mm256_permute4x64_epi64(
            _mm256_packus_epi16(
                _mm256_and_si256(low16, mask),
                _mm256_and_si256(high16, mask)),
            0b11011000) };
    }
#endif
    template<int shift_amount> requires(shift_amount >= 0 && shift_amount <= 15)
        sint16x16 shift_right(sint16x16 input) { return sint16x16{ _mm256_srai_epi16(input.data, shift_amount) }; }

    template<int shift_amount> requires(shift_amount >= 0 && shift_amount <= 31)
        sint32x8 shift_right(sint32x8 input) { return sint32x8{ _mm256_srai_epi32(input.data, shift_amount) }; }

    template<int shift_amount> requires(shift_amount >= 0 && shift_amount <= 63)
        sint64x4 shift_right(sint64x4 input) { return sint64x4{ _mm256_srai_epi64(input.data, shift_amount) }; }
}

namespace fyx::simd  // left shift by runtime variable shift amount, logical for both signed and unsigned
{
#if defined(_FOYE_SIMD_ENABLE_EMULATED_)
    uint16x8 shift_left(uint16x8 input, uint16x8 shift_amount)
    {
        __m256i temp32 = _mm256_cvtepu16_epi32(input.data);
        __m256i temp_shift_amount32 = _mm256_cvtepu16_epi32(shift_amount.data);
        __m256i res = _mm256_sllv_epi32(temp32, temp_shift_amount32);

        __m256i shifted = _mm256_srli_epi32(res, 16);
        __m256i masked = _mm256_and_si256(res, _mm256_set1_epi32(0xFFFF));
        __m128i low_mask = _mm256_extractf128_si256(masked, 0);
        __m128i high_mask = _mm256_extractf128_si256(masked, 1);
        return uint16x8{ _mm_packus_epi32(low_mask, high_mask) };
    }

    uint8x16 shift_left(uint8x16 input, uint8x16 shift_amount)
    {
        uint16x8 res_low = fyx::simd::shift_left(
            fyx::simd::expand_low<uint16x8>(input),
            fyx::simd::expand_low<uint16x8>(shift_amount)
        );

        uint16x8 res_high = fyx::simd::shift_left(
            fyx::simd::expand_high<uint16x8>(input),
            fyx::simd::expand_high<uint16x8>(shift_amount)
        );

        uint16x16 res = fyx::simd::merge(res_low, res_high);
        return fyx::simd::narrowing<uint8x16>(res);
    }

#endif
    uint32x4 shift_left(uint32x4 input, uint32x4 shift_amount)
    {
        return uint32x4{ _mm_sllv_epi32(input.data, shift_amount.data) };
    }

    uint64x2 shift_left(uint64x2 input, uint64x2 shift_amount)
    {
        return uint64x2{ _mm_sllv_epi64(input.data, shift_amount.data) };
    }

#if defined(_FOYE_SIMD_ENABLE_EMULATED_)
    uint16x16 shift_left(uint16x16 input, uint16x16 shift_amount)
    {
        uint16x8 res_low = fyx::simd::shift_left(input.low_part(), 
            shift_amount.low_part());

        uint16x8 res_high = fyx::simd::shift_left(input.high_part(), 
            shift_amount.high_part());
        return fyx::simd::merge(res_low, res_high);
    }

    uint8x32 shift_left(uint8x32 input, uint8x32 shift_amount)
    {
        uint16x16 res_low = fyx::simd::shift_left(
            fyx::simd::expand_low<uint16x16>(input),
            fyx::simd::expand_low<as_unsigned_type<uint16x16>>(shift_amount)
        );

        uint16x16 res_high = fyx::simd::shift_left(
            fyx::simd::expand_high<uint16x16>(input),
            fyx::simd::expand_high<uint16x16>(shift_amount)
        );

        uint8x16 res_low8 = fyx::simd::narrowing<uint8x16>(res_low);
        uint8x16 res_high8 = fyx::simd::narrowing<uint8x16>(res_high);
        return fyx::simd::merge(res_low8, res_high8);
    }

#endif
    uint32x8 shift_left(uint32x8 input, uint32x8 shift_amount)
    {
        return uint32x8{ _mm256_sllv_epi32(input.data, shift_amount.data) };
    }

    uint64x4 shift_left(uint64x4 input, uint64x4 shift_amount)
    {
        return uint64x4{ _mm256_sllv_epi64(input.data, shift_amount.data) };
    }

#if defined(_FOYE_SIMD_ENABLE_EMULATED_)
    sint8x16 shift_left(sint8x16 input, sint8x16 shift_amount)
    {
        return fyx::simd::reinterpret<sint8x16>(
            fyx::simd::shift_left(
                fyx::simd::reinterpret<uint8x16>(input), 
                fyx::simd::reinterpret<uint8x16>(shift_amount))
        );
    }

    sint16x8 shift_left(sint16x8 input, sint16x8 shift_amount)
    {
        return fyx::simd::reinterpret<sint16x8>(
            fyx::simd::shift_left(
                fyx::simd::reinterpret<uint16x8>(input), 
                fyx::simd::reinterpret<uint16x8>(shift_amount))
        );
    }
#endif
    sint32x4 shift_left(sint32x4 input, sint32x4 shift_amount)
    {
        return sint32x4{ _mm_sllv_epi32(input.data, shift_amount.data) };
    }

    sint64x2 shift_left(sint64x2 input, sint64x2 shift_amount)
    {
        return sint64x2{ _mm_sllv_epi64(input.data, shift_amount.data) };
    }

#if defined(_FOYE_SIMD_ENABLE_EMULATED_)
    sint8x32 shift_left(sint8x32 input, sint8x32 shift_amount)
    {
        return fyx::simd::reinterpret<sint8x32>(
            fyx::simd::shift_left(
                fyx::simd::reinterpret<uint8x32>(input), 
                fyx::simd::reinterpret<uint8x32>(shift_amount))
        );
    }

    sint16x16 shift_left(sint16x16 input, sint16x16 shift_amount)
    {
        return fyx::simd::reinterpret<sint16x16>(
            fyx::simd::shift_left(
                fyx::simd::reinterpret<uint16x16>(input), 
                fyx::simd::reinterpret<uint16x16>(shift_amount))
        );
    }
#endif
    sint32x8 shift_left(sint32x8 input, sint32x8 shift_amount)
    {
        return sint32x8{ _mm256_sllv_epi32(input.data, shift_amount.data) };
    }

    sint64x4 shift_left(sint64x4 input, sint64x4 shift_amount)
    {
        return sint64x4{ _mm256_sllv_epi64(input.data, shift_amount.data) };
    }
}


namespace fyx::simd // right shift by runtime variable shift amount, arithmetic for signed and logical for unsigned
{
#if defined(_FOYE_SIMD_ENABLE_EMULATED_)
    uint16x8 shift_right(uint16x8 input, uint16x8 shift_amount)
    {
        __m256i temp32 = _mm256_cvtepu16_epi32(input.data);
        __m256i temp_shift_amount32 = _mm256_cvtepu16_epi32(shift_amount.data);
        __m256i res = _mm256_srlv_epi32(temp32, temp_shift_amount32);

        __m256i shifted = _mm256_srli_epi32(res, 16);
        __m256i masked = _mm256_and_si256(res, _mm256_set1_epi32(0xFFFF));
        __m128i low_mask = _mm256_extractf128_si256(masked, 0);
        __m128i high_mask = _mm256_extractf128_si256(masked, 1);
        return uint16x8{ _mm_packus_epi32(low_mask, high_mask) };
    }

    uint8x16 shift_right(uint8x16 input, uint8x16 shift_amount)
    {
        uint16x8 res_low = fyx::simd::shift_right(
            fyx::simd::expand_low<uint16x8>(input),
            fyx::simd::expand_low<uint16x8>(shift_amount)
        );

        uint16x8 res_high = fyx::simd::shift_right(
            fyx::simd::expand_high<uint16x8>(input),
            fyx::simd::expand_high<uint16x8>(shift_amount)
        );

        uint16x16 res = fyx::simd::merge(res_low, res_high);
        return fyx::simd::narrowing<uint8x16>(res);
    }

#endif
    uint32x4 shift_right(uint32x4 input, uint32x4 shift_amount)
    {
        return uint32x4{ _mm_srlv_epi32(input.data, shift_amount.data) };
    }

    uint64x2 shift_right(uint64x2 input, uint64x2 shift_amount)
    {
        return uint64x2{ _mm_srlv_epi64(input.data, shift_amount.data) };
    }

#if defined(_FOYE_SIMD_ENABLE_EMULATED_)
    uint16x16 shift_right(uint16x16 input, uint16x16 shift_amount)
    {
        uint16x8 res_low = fyx::simd::shift_right(input.low_part(), shift_amount.low_part());
        uint16x8 res_high = fyx::simd::shift_right(input.high_part(), shift_amount.high_part());
        return fyx::simd::merge(res_low, res_high);
    }

    uint8x32 shift_right(uint8x32 input, uint8x32 shift_amount)
    {
        uint16x16 res_low = fyx::simd::shift_right(
            fyx::simd::expand_low<uint16x16>(input),
            fyx::simd::expand_low<uint16x16>(shift_amount)
        );

        uint16x16 res_high = fyx::simd::shift_right(
            fyx::simd::expand_high<uint16x16>(input),
            fyx::simd::expand_high<uint16x16>(shift_amount)
        );

        uint8x16 res_low8 = fyx::simd::narrowing<uint8x16>(res_low);
        uint8x16 res_high8 = fyx::simd::narrowing<uint8x16>(res_high);
        return fyx::simd::merge(res_low8, res_high8);
    }

#endif
    uint32x8 shift_right(uint32x8 input, uint32x8 shift_amount)
    {
        return uint32x8{ _mm256_srlv_epi32(input.data, shift_amount.data) };
    }

    uint64x4 shift_right(uint64x4 input, uint64x4 shift_amount)
    {
        return uint64x4{ _mm256_srlv_epi64(input.data, shift_amount.data) };
    }


    sint16x8 shift_right(sint16x8 input, sint16x8 shift_amount)
    {
        return sint16x8{ _mm_srav_epi16(input.data, shift_amount.data) };
    }

    sint32x4 shift_right(sint32x4 input, sint32x4 shift_amount)
    {
        return sint32x4{ _mm_srav_epi32(input.data, shift_amount.data) };
    }

    sint64x2 shift_right(sint64x2 input, sint64x2 shift_amount)
    {
        return sint64x2{ _mm_srav_epi64(input.data, shift_amount.data) };
    }


    sint16x16 shift_right(sint16x16 input, sint16x16 shift_amount)
    {
        return sint16x16{ _mm256_srav_epi16(input.data, shift_amount.data) };
    }

    sint32x8 shift_right(sint32x8 input, sint32x8 shift_amount)
    {
        return sint32x8{ _mm256_srav_epi32(input.data, shift_amount.data) };
    }

    sint64x4 shift_right(sint64x4 input, sint64x4 shift_amount)
    {
        return sint64x4{ _mm256_srav_epi64(input.data, shift_amount.data) };
    }

#if defined(_FOYE_SIMD_ENABLE_EMULATED_)
    sint8x16 shift_right(sint8x16 input, sint8x16 shift_amount)
    {
        sint16x16 input16 = fyx::simd::expand<sint16x16>(input);
        sint16x16 shift_amount16 = fyx::simd::expand<sint16x16>(shift_amount);
        sint16x16 result16 = fyx::simd::shift_right(input16, shift_amount16);
        return fyx::simd::narrowing<sint8x16>(result16);
    }

    sint8x32 shift_right(sint8x32 input, sint8x32 shift_amount)
    {
        sint16x16 res_low = fyx::simd::shift_right(
            fyx::simd::expand_low<sint16x16>(input),
            fyx::simd::expand_low<sint16x16>(shift_amount)
        );

        sint16x16 res_high = fyx::simd::shift_right(
            fyx::simd::expand_high<sint16x16>(input),
            fyx::simd::expand_high<sint16x16>(shift_amount)
        );

        sint8x16 res_low8 = fyx::simd::narrowing<sint8x16>(res_low);
        sint8x16 res_high8 = fyx::simd::narrowing<sint8x16>(res_high);
        return fyx::simd::merge(res_low8, res_high8);
    }
#endif
}

#endif
