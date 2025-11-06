#ifndef _FOYE_SIMD_REDUCE_HPP_
#define _FOYE_SIMD_REDUCE_HPP_
#pragma once

#include "simd_def.hpp"
#include "simd_opt.hpp"

namespace fyx::simd
{
    std::int64_t hsum(sint8x16 input)
    {
        const __m128i vzero = _mm_setzero_si128();
        __m128i mapped = _mm_xor_si128(input.data,
            _mm_set1_epi8(static_cast<char>(-128)));

        __m128i sum128 = _mm_sad_epu8(mapped, vzero);
        return (_mm_cvtsi128_si64(sum128) + _mm_extract_epi64(sum128, 1)) - 2048;
    }

    std::int64_t hsum(sint16x8 input)
    {
        __m128i sum32 = _mm_add_epi32(
            _mm_cvtepi16_epi32(input.data),
            _mm_cvtepi16_epi32(_mm_srli_si128(input.data, 8)));

        __m128i hadd1 = _mm_hadd_epi32(sum32, sum32);
        __m128i hadd2 = _mm_hadd_epi32(hadd1, hadd1);
        return static_cast<std::int64_t>(_mm_cvtsi128_si32(hadd2));
    }

    std::int64_t hsum(sint32x4 input)
    {
        __m128i sum64 = _mm_add_epi64(
            _mm_cvtepi32_epi64(input.data),
            _mm_cvtepi32_epi64(_mm_srli_si128(input.data, 8)));

        return _mm_extract_epi64(sum64, 0) + _mm_extract_epi64(sum64, 1);
    }

    std::int64_t hsum(sint64x2 input)
    {
        return _mm_extract_epi64(input.data, 0) + _mm_extract_epi64(input.data, 1);
    }

    std::int64_t hsum(sint8x32 input)
    {
        const __m256i vzero = _mm256_setzero_si256();
        __m256i vmask = _mm256_set1_epi8(static_cast<std::int8_t>(-128));
        __m256i half = _mm256_sad_epu8(_mm256_xor_si256(input.data, vmask), vzero);
        __m128i quarter = _mm_add_epi32(
            _mm256_castsi256_si128(half),
            _mm256_extracti128_si256(half, 1));

        return (_mm_cvtsi128_si32(
            _mm_add_epi32(quarter,
                _mm_unpackhi_epi64(quarter, quarter)))) - 4096;
    }

    std::int64_t hsum(sint16x16 input)
    {
        __m256i vsrc = _mm256_add_epi32(
            _mm256_cvtepi16_epi32(_mm256_castsi256_si128(input.data)),
            _mm256_cvtepi16_epi32(_mm256_extracti128_si256(input.data, 1)));

        __m256i s0 = _mm256_hadd_epi32(vsrc, vsrc);
        s0 = _mm256_hadd_epi32(s0, s0);

        __m128i s1 = _mm256_extracti128_si256(s0, 1);
        s1 = _mm_add_epi32(_mm256_castsi256_si128(s0), s1);
        return static_cast<std::int64_t>(_mm_cvtsi128_si32(s1));
    }

    std::int64_t hsum(sint32x8 input)
    {
        __m256i s0 = _mm256_hadd_epi32(input.data, input.data);
        s0 = _mm256_hadd_epi32(s0, s0);

        __m128i s1 = _mm256_extracti128_si256(s0, 1);
        s1 = _mm_add_epi32(_mm256_castsi256_si128(s0), s1);
        return static_cast<std::int64_t>(_mm_cvtsi128_si32(s1));
    }

    std::int64_t hsum(sint64x4 input)
    {
        alignas(32) std::int64_t idx[2];
        _mm_store_si128(reinterpret_cast<__m128i*>(idx),
            _mm_add_epi64(
                _mm256_castsi256_si128(input.data),
                _mm256_extracti128_si256(input.data, 1)));
        return idx[0] + idx[1];
    }

    std::uint64_t hsum(uint8x16 input)
    {
        const __m128i vzero = _mm_setzero_si128();
        __m128i sum128 = _mm_sad_epu8(input.data, vzero);
        return _mm_cvtsi128_si64(sum128) + _mm_extract_epi64(sum128, 1);
    }

    std::uint64_t hsum(uint16x8 input)
    {
        __m128i sum32 = _mm_add_epi32(
            _mm_cvtepu16_epi32(input.data),
            _mm_cvtepu16_epi32(_mm_srli_si128(input.data, 8))
        );

        __m128i hadd1 = _mm_hadd_epi32(sum32, sum32);
        __m128i hadd2 = _mm_hadd_epi32(hadd1, hadd1);
        return static_cast<std::uint64_t>(_mm_cvtsi128_si32(hadd2));
    }

    std::uint64_t hsum(uint32x4 input)
    {
        __m128i sum64 = _mm_add_epi64(
            _mm_cvtepu32_epi64(input.data),
            _mm_cvtepu32_epi64(_mm_srli_si128(input.data, 8))
        );
        return _mm_extract_epi64(sum64, 0) + _mm_extract_epi64(sum64, 1);
    }

    std::uint64_t hsum(uint64x2 input)
    {
        return _mm_extract_epi64(input.data, 0) + _mm_extract_epi64(input.data, 1);
    }

    std::uint64_t hsum(uint8x32 input)
    {
        const __m256i vzero = _mm256_setzero_si256();
        __m256i half = _mm256_sad_epu8(input.data, vzero);
        __m128i quarter = _mm_add_epi32(
            _mm256_castsi256_si128(half),
            _mm256_extracti128_si256(half, 1));

        return _mm_cvtsi128_si32(
            _mm_add_epi32(quarter,
                _mm_unpackhi_epi64(quarter, quarter)));
    }

    std::uint64_t hsum(uint16x16 input)
    {
        __m256i vsrc = _mm256_add_epi32(
            _mm256_cvtepu16_epi32(_mm256_castsi256_si128(input.data)),
            _mm256_cvtepu16_epi32(_mm256_extracti128_si256(input.data, 1)));

        __m256i s0 = _mm256_hadd_epi32(vsrc, vsrc);
        s0 = _mm256_hadd_epi32(s0, s0);

        __m128i s1 = _mm256_extracti128_si256(s0, 1);
        s1 = _mm_add_epi32(_mm256_castsi256_si128(s0), s1);
        return static_cast<std::uint64_t>(_mm_cvtsi128_si32(s1));
    }

    std::uint64_t hsum(uint32x8 input)
    {
        return static_cast<std::uint64_t>(hsum(
            fyx::simd::reinterpret<sint32x8>(input)));
    }

    std::uint64_t hsum(uint64x4 input)
    {
        alignas(32) std::uint64_t idx[2];
        _mm_store_si128(reinterpret_cast<__m128i*>(idx),
            _mm_add_epi64(
                _mm256_castsi256_si128(input.data),
                _mm256_extracti128_si256(input.data, 1)));
        return idx[0] + idx[1];
    }

    float hsum(float32x4 input)
    {
        __m128 v = input.data;
        v = _mm_hadd_ps(v, v);
        v = _mm_hadd_ps(v, v);
        return _mm_cvtss_f32(v);
    }

    double hsum(float64x2 input)
    {
        __m128d v = input.data;
        v = _mm_hadd_pd(v, v);
        return _mm_cvtsd_f64(v);
    }

    float hsum(float32x8 input)
    {
        __m128 sum128 = _mm_add_ps(
            _mm256_castps256_ps128(input.data),
            _mm256_extractf128_ps(input.data, 1));

        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        return _mm_cvtss_f32(sum128);
    }

    double hsum(float64x4 input)
    {
        __m256d s0 = _mm256_hadd_pd(input.data, input.data);
        return static_cast<double>(_mm_cvtsd_f64(
            _mm_add_pd(
                _mm256_castpd256_pd128(s0),
                _mm256_extractf128_pd(s0, 1))));
    }

    uint8x16::scalar_t hmin(uint8x16 input)
    {
        __m128i val = input.data;
        val = _mm_min_epu8(val, _mm_srli_si128(val, 8));
        val = _mm_min_epu8(val, _mm_srli_si128(val, 4));
        val = _mm_min_epu8(val, _mm_srli_si128(val, 2));
        val = _mm_min_epu8(val, _mm_srli_si128(val, 1));
        return static_cast<typename uint8x16::scalar_t>(_mm_cvtsi128_si32(val));
    }

    uint16x8::scalar_t hmin(uint16x8 input)
    {
        __m128i val = input.data;
        val = _mm_min_epu16(val, _mm_srli_si128(val, 8));
        val = _mm_min_epu16(val, _mm_srli_si128(val, 4));
        val = _mm_min_epu16(val, _mm_srli_si128(val, 2));
        return static_cast<typename uint16x8::scalar_t>(_mm_cvtsi128_si32(val));
    }

    uint32x4::scalar_t hmin(uint32x4 input)
    {
        __m128i val = input.data;
        val = _mm_min_epu32(val, _mm_srli_si128(val, 8));
        val = _mm_min_epu32(val, _mm_srli_si128(val, 4));
        return static_cast<typename uint32x4::scalar_t>(_mm_cvtsi128_si32(val));
    }

    sint8x16::scalar_t hmin(sint8x16 input)
    {
        __m128i val = input.data;
        val = _mm_min_epi8(val, _mm_srli_si128(val, 8));
        val = _mm_min_epi8(val, _mm_srli_si128(val, 4));
        val = _mm_min_epi8(val, _mm_srli_si128(val, 2));
        val = _mm_min_epi8(val, _mm_srli_si128(val, 1));
        return static_cast<typename sint8x16::scalar_t>(_mm_extract_epi8(val, 0));
    }

    sint16x8::scalar_t hmin(sint16x8 input)
    {
        __m128i val = input.data;
        val = _mm_min_epi16(val, _mm_srli_si128(val, 8));
        val = _mm_min_epi16(val, _mm_srli_si128(val, 4));
        val = _mm_min_epi16(val, _mm_srli_si128(val, 2));
        return static_cast<typename sint16x8::scalar_t>(_mm_extract_epi16(val, 0));
    }

    sint32x4::scalar_t hmin(sint32x4 input)
    {
        __m128i val = input.data;
        val = _mm_min_epi32(val, _mm_srli_si128(val, 8));
        val = _mm_min_epi32(val, _mm_srli_si128(val, 4));
        return static_cast<typename sint32x4::scalar_t>(_mm_cvtsi128_si32(val));
    }

    uint8x32::scalar_t hmin(uint8x32 input)
    {
        __m128i val = _mm_min_epu8(_mm256_castsi256_si128(input.data),
            _mm256_extracti128_si256(input.data, 1));
        val = _mm_min_epu8(val, _mm_srli_si128(val, 8));
        val = _mm_min_epu8(val, _mm_srli_si128(val, 4));
        val = _mm_min_epu8(val, _mm_srli_si128(val, 2));
        val = _mm_min_epu8(val, _mm_srli_si128(val, 1));
        return static_cast<typename uint8x32::scalar_t>(_mm_cvtsi128_si32(val));
    }

    uint16x16::scalar_t hmin(uint16x16 input)
    {
        __m128i v0 = _mm256_castsi256_si128(input.data);
        v0 = _mm_min_epu16(v0, _mm256_extracti128_si256(input.data, 1));
        v0 = _mm_min_epu16(v0, _mm_srli_si128(v0, 8));
        v0 = _mm_min_epu16(v0, _mm_srli_si128(v0, 4));
        v0 = _mm_min_epu16(v0, _mm_srli_si128(v0, 2));
        return static_cast<typename uint16x16::scalar_t>(_mm_cvtsi128_si32(v0));
    }

    uint32x8::scalar_t hmin(uint32x8 input)
    {
        __m128i v0 = _mm256_castsi256_si128(input.data);
        __m128i v1 = _mm256_extracti128_si256(input.data, 1);
        v0 = _mm_min_epu32(v0, v1);
        v0 = _mm_min_epu32(v0, _mm_srli_si128(v0, 8));
        v0 = _mm_min_epu32(v0, _mm_srli_si128(v0, 4));
        return static_cast<typename uint32x8::scalar_t>(_mm_cvtsi128_si32(v0));
    }

    sint8x32::scalar_t hmin(sint8x32 input)
    {
        __m128i val = _mm_min_epi8(_mm256_castsi256_si128(input.data),
            _mm256_extracti128_si256(input.data, 1));
        val = _mm_min_epi8(val, _mm_srli_si128(val, 8));
        val = _mm_min_epi8(val, _mm_srli_si128(val, 4));
        val = _mm_min_epi8(val, _mm_srli_si128(val, 2));
        val = _mm_min_epi8(val, _mm_srli_si128(val, 1));
        return static_cast<typename sint8x32::scalar_t>(_mm_cvtsi128_si32(val));
    }

    sint16x16::scalar_t hmin(sint16x16 input)
    {
        __m128i v0 = _mm256_castsi256_si128(input.data);
        __m128i v1 = _mm256_extracti128_si256(input.data, 1);
        v0 = _mm_min_epi16(v0, v1);
        v0 = _mm_min_epi16(v0, _mm_srli_si128(v0, 8));
        v0 = _mm_min_epi16(v0, _mm_srli_si128(v0, 4));
        v0 = _mm_min_epi16(v0, _mm_srli_si128(v0, 2));
        return static_cast<typename sint16x16::scalar_t>(_mm_cvtsi128_si32(v0));
    }

    sint32x8::scalar_t hmin(sint32x8 input)
    {
        __m128i v0 = _mm256_castsi256_si128(input.data);
        __m128i v1 = _mm256_extracti128_si256(input.data, 1);
        v0 = _mm_min_epi32(v0, v1);
        v0 = _mm_min_epi32(v0, _mm_srli_si128(v0, 8));
        v0 = _mm_min_epi32(v0, _mm_srli_si128(v0, 4));
        return static_cast<typename sint32x8::scalar_t>(_mm_cvtsi128_si32(v0));
    }

    float32x4::scalar_t hmin(float32x4 input)
    {
        __m128 v0 = input.data;
        v0 = _mm_min_ps(v0, _mm_movehl_ps(v0, v0));
        v0 = _mm_min_ss(v0, _mm_shuffle_ps(v0, v0, 1));
        return _mm_cvtss_f32(v0);
    }

    float32x8::scalar_t hmin(float32x8 input)
    {
        __m128 v0 = _mm256_castps256_ps128(input.data);
        __m128 v1 = _mm256_extractf128_ps(input.data, 1);
        v0 = _mm_min_ps(v0, v1);
        v0 = _mm_min_ps(v0, _mm_permute_ps(v0, (((0) << 6) | ((0) << 4) | ((3) << 2) | ((2)))));
        v0 = _mm_min_ps(v0, _mm_permute_ps(v0, (((0) << 6) | ((0) << 4) | ((0) << 2) | ((1)))));
        return static_cast<typename float32x8::scalar_t>(_mm_cvtss_f32(v0));
    }

    float64x4::scalar_t hmin(float64x4 input)
    {
        __m256d v0 = input.data;
        __m128d v1 = _mm256_extractf128_pd(v0, 1);
        __m128d v2 = _mm256_castpd256_pd128(v0);
        v2 = _mm_min_pd(v2, v1);
        v2 = _mm_min_pd(v2, _mm_permute_pd(v2, 1));
        return static_cast<typename float64x4::scalar_t>(_mm_cvtsd_f64(v2));
    }


    uint8x16::scalar_t hmax(uint8x16 input)
    {
        __m128i v = input.data;
        v = _mm_max_epu8(v, _mm_srli_si128(v, 8));
        v = _mm_max_epu8(v, _mm_srli_si128(v, 4));
        v = _mm_max_epu8(v, _mm_srli_si128(v, 2));
        v = _mm_max_epu8(v, _mm_srli_si128(v, 1));
        return static_cast<typename uint8x16::scalar_t>(_mm_cvtsi128_si32(v));
    }

    uint16x8::scalar_t hmax(uint16x8 input)
    {
        __m128i v = input.data;
        v = _mm_max_epu16(v, _mm_srli_si128(v, 8));
        v = _mm_max_epu16(v, _mm_srli_si128(v, 4));
        v = _mm_max_epu16(v, _mm_srli_si128(v, 2));
        return static_cast<typename uint16x8::scalar_t>(_mm_extract_epi16(v, 0));
    }

    uint32x4::scalar_t hmax(uint32x4 input)
    {
        __m128i v = input.data;
        v = _mm_max_epu32(v, _mm_srli_si128(v, 8));
        v = _mm_max_epu32(v, _mm_srli_si128(v, 4));
        return static_cast<typename uint32x4::scalar_t>(_mm_cvtsi128_si32(v));
    }

    sint8x16::scalar_t hmax(sint8x16 input)
    {
        __m128i v = input.data;
        v = _mm_max_epi8(v, _mm_srli_si128(v, 8));
        v = _mm_max_epi8(v, _mm_srli_si128(v, 4));
        v = _mm_max_epi8(v, _mm_srli_si128(v, 2));
        v = _mm_max_epi8(v, _mm_srli_si128(v, 1));
        return static_cast<typename sint8x16::scalar_t>(_mm_cvtsi128_si32(v));
    }

    sint16x8::scalar_t hmax(sint16x8 input)
    {
        __m128i v = input.data;
        v = _mm_max_epi16(v, _mm_srli_si128(v, 8));
        v = _mm_max_epi16(v, _mm_srli_si128(v, 4));
        v = _mm_max_epi16(v, _mm_srli_si128(v, 2));
        return static_cast<typename sint16x8::scalar_t>(_mm_extract_epi16(v, 0));
    }

    sint32x4::scalar_t hmax(sint32x4 input)
    {
        __m128i v = input.data;
        v = _mm_max_epi32(v, _mm_srli_si128(v, 8));
        v = _mm_max_epi32(v, _mm_srli_si128(v, 4));
        return static_cast<typename sint32x4::scalar_t>(_mm_cvtsi128_si32(v));
    }

    uint8x32::scalar_t hmax(uint8x32 input)
    {
        __m128i val = _mm_max_epu8((_mm256_castsi256_si128(input.data)),
            (_mm256_extracti128_si256((input.data), 1)));
        val = _mm_max_epu8(val, _mm_srli_si128(val, 8));
        val = _mm_max_epu8(val, _mm_srli_si128(val, 4));
        val = _mm_max_epu8(val, _mm_srli_si128(val, 2));
        val = _mm_max_epu8(val, _mm_srli_si128(val, 1));
        return static_cast<typename uint8x32::scalar_t>(_mm_cvtsi128_si32(val));
    }

    uint16x16::scalar_t hmax(uint16x16 input)
    {
        __m128i v0 = (_mm256_castsi256_si128(input.data));
        __m128i v1 = (_mm256_extracti128_si256((input.data), 1));
        v0 = _mm_max_epu16(v0, v1);
        v0 = _mm_max_epu16(v0, _mm_srli_si128(v0, 8));
        v0 = _mm_max_epu16(v0, _mm_srli_si128(v0, 4));
        v0 = _mm_max_epu16(v0, _mm_srli_si128(v0, 2));
        return static_cast<typename uint16x16::scalar_t>(_mm_cvtsi128_si32(v0));
    }

    uint32x8::scalar_t hmax(uint32x8 input)
    {
        __m128i v0 = (_mm256_castsi256_si128(input.data));
        __m128i v1 = (_mm256_extracti128_si256((input.data), 1));
        v0 = _mm_max_epu32(v0, v1);
        v0 = _mm_max_epu32(v0, _mm_srli_si128(v0, 8));
        v0 = _mm_max_epu32(v0, _mm_srli_si128(v0, 4));
        return static_cast<typename uint32x8::scalar_t>(_mm_cvtsi128_si32(v0));
    }

    sint8x32::scalar_t hmax(sint8x32 input)
    {
        __m128i val = _mm_max_epi8((_mm256_castsi256_si128(input.data)),
            (_mm256_extracti128_si256((input.data), 1)));
        val = _mm_max_epi8(val, _mm_srli_si128(val, 8));
        val = _mm_max_epi8(val, _mm_srli_si128(val, 4));
        val = _mm_max_epi8(val, _mm_srli_si128(val, 2));
        val = _mm_max_epi8(val, _mm_srli_si128(val, 1));
        return static_cast<typename sint8x32::scalar_t>(_mm_cvtsi128_si32(val));
    }

    sint16x16::scalar_t hmax(sint16x16 input)
    {
        __m128i v0 = (_mm256_castsi256_si128(input.data));
        __m128i v1 = (_mm256_extracti128_si256((input.data), 1));
        v0 = _mm_max_epi16(v0, v1);
        v0 = _mm_max_epi16(v0, _mm_srli_si128(v0, 8));
        v0 = _mm_max_epi16(v0, _mm_srli_si128(v0, 4));
        v0 = _mm_max_epi16(v0, _mm_srli_si128(v0, 2));
        return static_cast<typename sint16x16::scalar_t>(_mm_cvtsi128_si32(v0));
    }

    sint32x8::scalar_t hmax(sint32x8 input)
    {
        __m128i v0 = (_mm256_castsi256_si128(input.data));
        __m128i v1 = (_mm256_extracti128_si256((input.data), 1));
        v0 = _mm_max_epi32(v0, v1);
        v0 = _mm_max_epi32(v0, _mm_srli_si128(v0, 8));
        v0 = _mm_max_epi32(v0, _mm_srli_si128(v0, 4));
        return static_cast<typename sint32x8::scalar_t>(_mm_cvtsi128_si32(v0));
    }

    float32x4::scalar_t hmax(float32x4 input)
    {
        __m128 v = input.data;
        __m128 high = _mm_movehl_ps(v, v);
        __m128 max1 = _mm_max_ps(v, high);
        __m128 max2 = _mm_max_ss(max1, _mm_shuffle_ps(max1, max1, _MM_SHUFFLE(0, 0, 0, 1)));
        return _mm_cvtss_f32(max2);
    }

    float32x8::scalar_t hmax(float32x8 input)
    {
        __m128 v0 = (_mm256_castps256_ps128(input.data));
        __m128 v1 = (_mm256_extractf128_ps((input.data), 1));
        v0 = _mm_max_ps(v0, v1);
        v0 = _mm_max_ps(v0, _mm_permute_ps(v0, (((0) << 6) | ((0) << 4) | ((3) << 2) | ((2)))));
        v0 = _mm_max_ps(v0, _mm_permute_ps(v0, (((0) << 6) | ((0) << 4) | ((0) << 2) | ((1)))));
        return _mm_cvtss_f32(v0);
    }

    float64x4::scalar_t hmax(float64x4 input)
    {
        __m256d v0 = input.data;
        __m128d v1 = _mm256_extractf128_pd(v0, 1);
        __m128d v2 = _mm256_castpd256_pd128(v0);
        v2 = _mm_max_pd(v2, v1);
        v2 = _mm_max_pd(v2, _mm_permute_pd(v2, 1));
        return static_cast<typename float64x4::scalar_t>(_mm_cvtsd_f64(v2));
    }

#define DEFINE_HMINHMAX_FALLBACK(input_simd_type, cmpfunc)\
input_simd_type::scalar_t h##cmpfunc##(input_simd_type input)\
{\
    alignas(alignof(typename input_simd_type::vector_t))\
        typename input_simd_type::scalar_t temp[input_simd_type::lane_width];\
    fyx::simd::store_aligned(input, temp);\
    return *std::##cmpfunc##_element(std::begin(temp), std::end(temp));\
}
#define DEFINE_HMINHMAX_FALLBACK_2WAY_DISPATCH(input_simd_type) \
    DEFINE_HMINHMAX_FALLBACK(input_simd_type, min) DEFINE_HMINHMAX_FALLBACK(input_simd_type, max)

    DEFINE_HMINHMAX_FALLBACK_2WAY_DISPATCH(uint64x2)
    DEFINE_HMINHMAX_FALLBACK_2WAY_DISPATCH(sint64x2)
    DEFINE_HMINHMAX_FALLBACK_2WAY_DISPATCH(uint64x4)
    DEFINE_HMINHMAX_FALLBACK_2WAY_DISPATCH(sint64x4)
    DEFINE_HMINHMAX_FALLBACK_2WAY_DISPATCH(float64x2)
#undef DEFINE_HMINHMAX_FALLBACK_2WAY_DISPATCH
#undef DEFINE_HMINHMAX_FALLBACK
}

namespace fyx::simd
{
#define DEF_NOSUITABLE_IMPLEMENT__HMUL(funcd) \
template<typename T = void> \
funcd \
{ \
    static_assert(fyx::simd::detail::dependent_false<T>::value, \
        "There is no suitable instruction combination to achieve this function: " #funcd \
        ". because determining the bit width of a multiplication accumulator without knowing its purpose is reckless"); \
}
    DEF_NOSUITABLE_IMPLEMENT__HMUL(std::uint64_t hmul(uint8x16 input))
    DEF_NOSUITABLE_IMPLEMENT__HMUL(std::uint64_t hmul(uint16x8 input))
    DEF_NOSUITABLE_IMPLEMENT__HMUL(std::uint64_t hmul(uint32x4 input))
    DEF_NOSUITABLE_IMPLEMENT__HMUL(std::uint64_t hmul(uint64x2 input))
    DEF_NOSUITABLE_IMPLEMENT__HMUL(std::int64_t hmul(sint8x16 input))
    DEF_NOSUITABLE_IMPLEMENT__HMUL(std::int64_t hmul(sint16x8 input))
    DEF_NOSUITABLE_IMPLEMENT__HMUL(std::int64_t hmul(sint32x4 input))
    DEF_NOSUITABLE_IMPLEMENT__HMUL(std::int64_t hmul(sint64x2 input))
    DEF_NOSUITABLE_IMPLEMENT__HMUL(std::uint64_t hmul(uint8x32 input))
    DEF_NOSUITABLE_IMPLEMENT__HMUL(std::uint64_t hmul(uint16x16 input))
    DEF_NOSUITABLE_IMPLEMENT__HMUL(std::uint64_t hmul(uint32x8 input))
    DEF_NOSUITABLE_IMPLEMENT__HMUL(std::uint64_t hmul(uint64x4 input))
    DEF_NOSUITABLE_IMPLEMENT__HMUL(std::int64_t hmul(sint8x32 input))
    DEF_NOSUITABLE_IMPLEMENT__HMUL(std::int64_t hmul(sint16x16 input))
    DEF_NOSUITABLE_IMPLEMENT__HMUL(std::int64_t hmul(sint32x8 input))
    DEF_NOSUITABLE_IMPLEMENT__HMUL(std::int64_t hmul(sint64x4 input))
#undef DEF_NOSUITABLE_IMPLEMENT__HMUL

    float hmul(float32x4 input)
    {
        __m128 v = input.data;
        __m128 prod = _mm_mul_ps(v, _mm_movehdup_ps(v));

        float prod1 = _mm_cvtss_f32(prod);
        float prod2 = _mm_cvtss_f32(_mm_movehl_ps(prod, prod));

        return prod1 * prod2;
    }

    double hmul(float64x2 input)
    {
        __m128d v = input.data;
        __m128d swapped = _mm_shuffle_pd(v, v, 0x1);
        __m128d result = _mm_mul_pd(v, swapped);
        return _mm_cvtsd_f64(result);
    }

    float hmul(float32x8 input)
    {
        __m128 v0 = _mm256_castps256_ps128(input.data);
        __m128 v1 = _mm256_extractf128_ps(input.data, 1);

        __m128 t0 = _mm_mul_ps(v0, _mm_movehl_ps(v0, v0));
        t0 = _mm_mul_ps(t0, _mm_permute_ps(t0, _MM_SHUFFLE(1, 1, 1, 1)));
        float prod0 = _mm_cvtss_f32(t0);

        __m128 t1 = _mm_mul_ps(v1, _mm_movehl_ps(v1, v1));
        t1 = _mm_mul_ps(t1, _mm_permute_ps(t1, _MM_SHUFFLE(1, 1, 1, 1)));
        float prod1 = _mm_cvtss_f32(t1);

        return prod0 * prod1;
    }

    double hmul(float64x4 input)
    {
        __m256d v = input.data;
        __m256d swapped = _mm256_permute4x64_pd(v, _MM_SHUFFLE(1, 0, 3, 2));

        __m256d mul = _mm256_mul_pd(v, swapped);
        __m128d low = _mm256_castpd256_pd128(mul);
        return _mm_cvtsd_f64(low) * _mm_cvtsd_f64(_mm_permute_pd(low, 0b11));
    }
}

#endif
