#ifndef _FOYE_SIMD_CVT_HPP_
#define _FOYE_SIMD_CVT_HPP_
#pragma once

#include "simd_def.hpp"
#include "simd_utility.hpp"

namespace fyx::simd
{
	float32x8 floor(float32x8 input) { return float32x8{ _mm256_round_ps(input.data, _MM_FROUND_FLOOR) }; }
	float32x4 floor(float32x4 input) { return float32x4{ _mm_round_ps(input.data, _MM_FROUND_FLOOR) }; }
	float64x4 floor(float64x4 input) { return float64x4{ _mm256_round_pd(input.data, _MM_FROUND_FLOOR) }; }
	float64x2 floor(float64x2 input) { return float64x2{ _mm_round_pd(input.data, _MM_FROUND_FLOOR) }; }

	float32x8 ceil(float32x8 input) { return float32x8{ _mm256_round_ps(input.data, _MM_FROUND_CEIL) }; }
	float32x4 ceil(float32x4 input) { return float32x4{ _mm_round_ps(input.data, _MM_FROUND_CEIL) }; }
	float64x4 ceil(float64x4 input) { return float64x4{ _mm256_round_pd(input.data, _MM_FROUND_CEIL) }; }
	float64x2 ceil(float64x2 input) { return float64x2{ _mm_round_pd(input.data, _MM_FROUND_CEIL) }; }

	float32x8 round(float32x8 input) { return float32x8{ _mm256_round_ps(input.data, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC) }; }
	float32x4 round(float32x4 input) { return float32x4{ _mm_round_ps(input.data, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC) }; }
	float64x4 round(float64x4 input) { return float64x4{ _mm256_round_pd(input.data, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC) }; }
	float64x2 round(float64x2 input) { return float64x2{ _mm_round_pd(input.data, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC) }; }

	float32x8 rint(float32x8 input) { return float32x8{ _mm256_round_ps(input.data, _MM_FROUND_RINT) }; }
	float32x4 rint(float32x4 input) { return float32x4{ _mm_round_ps(input.data, _MM_FROUND_RINT) }; }
	float64x4 rint(float64x4 input) { return float64x4{ _mm256_round_pd(input.data, _MM_FROUND_RINT) }; }
	float64x2 rint(float64x2 input) { return float64x2{ _mm_round_pd(input.data, _MM_FROUND_RINT) }; }

	float32x8 nearbyint(float32x8 input) { return float32x8{ _mm256_round_ps(input.data, _MM_FROUND_NEARBYINT) }; }
	float32x4 nearbyint(float32x4 input) { return float32x4{ _mm_round_ps(input.data, _MM_FROUND_NEARBYINT) }; }
	float64x4 nearbyint(float64x4 input) { return float64x4{ _mm256_round_pd(input.data, _MM_FROUND_NEARBYINT) }; }
	float64x2 nearbyint(float64x2 input) { return float64x2{ _mm_round_pd(input.data, _MM_FROUND_NEARBYINT) }; }

#if !defined(FOYE_SIMD_ENABLE_SVML)
	float32x8 trunc(float32x8 input) { return float32x8{ _mm256_trunc_ps(input.data) }; }
	float32x4 trunc(float32x4 input) { return float32x4{ _mm_trunc_ps(input.data) }; }
	float64x4 trunc(float64x4 input) { return float64x4{ _mm256_trunc_pd(input.data) }; }
	float64x2 trunc(float64x2 input) { return float64x2{ _mm_trunc_pd(input.data) }; }
#else
	float32x8 trunc(float32x8 input) { return float32x8{ _mm256_round_ps(input.data, _MM_FROUND_TO_ZERO) }; }
	float32x4 trunc(float32x4 input) { return float32x4{ _mm_round_ps(input.data, _MM_FROUND_TO_ZERO) }; }
	float64x4 trunc(float64x4 input) { return float64x4{ _mm256_round_pd(input.data, _MM_FROUND_TO_ZERO) }; }
	float64x2 trunc(float64x2 input) { return float64x2{ _mm_round_pd(input.data, _MM_FROUND_TO_ZERO) }; }
#endif

	sint32x8 trunc_as_i(float32x8 input) { return sint32x8{ _mm256_cvtps_epi32(fyx::simd::trunc(input).data) }; }
	sint32x4 trunc_as_i(float32x4 input) { return sint32x4{ _mm_cvtps_epi32(fyx::simd::trunc(input).data) }; }
	sint64x4 trunc_as_i(float64x4 input) { return sint64x4{ _mm256_cvtpd_epi64(fyx::simd::trunc(input).data) }; }
	sint64x2 trunc_as_i(float64x2 input) { return sint64x2{ _mm_cvtpd_epi64(fyx::simd::trunc(input).data) }; }

	sint32x8 floor_as_i(float32x8 input) { return sint32x8{ _mm256_cvtps_epi32(fyx::simd::floor(input).data) }; }
	sint32x4 floor_as_i(float32x4 input) { return sint32x4{ _mm_cvtps_epi32(fyx::simd::floor(input).data) }; }
	sint64x4 floor_as_i(float64x4 input) { return sint64x4{ _mm256_cvtpd_epi64(fyx::simd::floor(input).data) }; }
	sint64x2 floor_as_i(float64x2 input) { return sint64x2{ _mm_cvtpd_epi64(fyx::simd::floor(input).data) }; }

	sint32x8 ceil_as_i(float32x8 input) { return sint32x8{ _mm256_cvtps_epi32(fyx::simd::ceil(input).data) }; }
	sint32x4 ceil_as_i(float32x4 input) { return sint32x4{ _mm_cvtps_epi32(fyx::simd::ceil(input).data) }; }
	sint64x4 ceil_as_i(float64x4 input) { return sint64x4{ _mm256_cvtpd_epi64(fyx::simd::ceil(input).data) }; }
	sint64x2 ceil_as_i(float64x2 input) { return sint64x2{ _mm_cvtpd_epi64(fyx::simd::ceil(input).data) }; }

	sint32x8 round_as_i(float32x8 input) { return sint32x8{ _mm256_cvtps_epi32(fyx::simd::round(input).data) }; }
	sint32x4 round_as_i(float32x4 input) { return sint32x4{ _mm_cvtps_epi32(fyx::simd::round(input).data) }; }
	sint64x4 round_as_i(float64x4 input) { return sint64x4{ _mm256_cvtpd_epi64(fyx::simd::round(input).data) }; }
	sint64x2 round_as_i(float64x2 input) { return sint64x2{ _mm_cvtpd_epi64(fyx::simd::round(input).data) }; }

	sint32x8 rint_as_i(float32x8 input) { return sint32x8{ _mm256_cvtps_epi32(fyx::simd::rint(input).data) }; }
	sint32x4 rint_as_i(float32x4 input) { return sint32x4{ _mm_cvtps_epi32(fyx::simd::rint(input).data) }; }
	sint64x4 rint_as_i(float64x4 input) { return sint64x4{ _mm256_cvtpd_epi64(fyx::simd::rint(input).data) }; }
	sint64x2 rint_as_i(float64x2 input) { return sint64x2{ _mm_cvtpd_epi64(fyx::simd::rint(input).data) }; }

	sint32x8 nearbyint_as_i(float32x8 input) { return sint32x8{ _mm256_cvtps_epi32(fyx::simd::nearbyint(input).data) }; }
	sint32x4 nearbyint_as_i(float32x4 input) { return sint32x4{ _mm_cvtps_epi32(fyx::simd::nearbyint(input).data) }; }
	sint64x4 nearbyint_as_i(float64x4 input) { return sint64x4{ _mm256_cvtpd_epi64(fyx::simd::nearbyint(input).data) }; }
	sint64x2 nearbyint_as_i(float64x2 input) { return sint64x2{ _mm_cvtpd_epi64(fyx::simd::nearbyint(input).data) }; }

	template<typename target_simd, typename source_simd> target_simd expand(source_simd) { __assume(false); }
	template<typename target_simd, typename source_simd> target_simd expand_low(source_simd) { __assume(false); }
	template<typename target_simd, typename source_simd> target_simd expand_high(source_simd) { __assume(false); }
	template<typename target_simd, typename source_simd> target_simd floating(source_simd) { __assume(false); }
	template<typename target_simd, typename source_simd> target_simd narrowing(source_simd) { __assume(false); }

	template<> uint16x16 expand<uint16x16, uint8x16>(uint8x16 input) { return uint16x16{ _mm256_cvtepu8_epi16(input.data) }; }
	template<> sint16x16 expand<sint16x16, uint8x16>(uint8x16 input) { return sint16x16{ _mm256_cvtepu8_epi16(input.data) }; }
	template<> uint32x8 expand<uint32x8, uint16x8>(uint16x8 input) { return uint32x8{ _mm256_cvtepu16_epi32(input.data) }; }
	template<> sint32x8 expand<sint32x8, uint16x8>(uint16x8 input) { return sint32x8{ _mm256_cvtepu16_epi32(input.data) }; }
	template<> uint64x4 expand<uint64x4, uint32x4>(uint32x4 input) { return uint64x4{ _mm256_cvtepu32_epi64(input.data) }; }
	template<> sint64x4 expand<sint64x4, uint32x4>(uint32x4 input) { return sint64x4{ _mm256_cvtepu32_epi64(input.data) }; }
	template<> sint16x16 expand<sint16x16, sint8x16>(sint8x16 input) { return sint16x16{ _mm256_cvtepi8_epi16(input.data) }; }
	template<> sint32x8 expand<sint32x8, sint16x8>(sint16x8 input) { return sint32x8{ _mm256_cvtepi16_epi32(input.data) }; }
	template<> sint64x4 expand<sint64x4, sint32x4>(sint32x4 input) { return sint64x4{ _mm256_cvtepi32_epi64(input.data) }; }
	template<> uint16x16 expand<uint16x16, sint8x16>(sint8x16 input) { return uint16x16{ _mm256_cvtepu8_epi16(input.data) }; }
	template<> uint32x8 expand<uint32x8, sint16x8>(sint16x8 input) { return uint32x8{ _mm256_cvtepu16_epi32(input.data) }; }
	template<> uint64x4 expand<uint64x4, sint32x4>(sint32x4 input) { return uint64x4{ _mm256_cvtepu32_epi64(input.data) }; }
	template<> float64x4 expand<float64x4, float32x4>(float32x4 input) { return float64x4{ _mm256_cvtps_pd(input.data) }; }
#if defined(_FOYE_SIMD_HAS_FP16_)
	template<> float32x8 expand<float32x8, float16x8>(float16x8 input) { return float32x8{ _mm256_cvtph_ps(input.data) }; }
#endif
#if defined(_FOYE_SIMD_HAS_BF16_)
	template<> float32x8 expand<float32x8, bfloat16x8>(bfloat16x8 input) 
	{
		return float32x8{ _mm256_castsi256_ps(
			_mm256_slli_epi32(
				_mm256_cvtepu16_epi32(input.data), 16)) };
	}
#endif
	
	template<> uint16x8 expand_low<uint16x8, uint8x16>(uint8x16 input) { return uint16x8{ _mm_cvtepu8_epi16(input.data) }; }
	template<> uint32x4 expand_low<uint32x4, uint16x8>(uint16x8 input) { return uint32x4{ _mm_cvtepu16_epi32(input.data) }; }
	template<> uint64x2 expand_low<uint64x2, uint32x4>(uint32x4 input) { return uint64x2{ _mm_cvtepu32_epi64(input.data) }; }
	template<> sint16x8 expand_low<sint16x8, uint8x16>(uint8x16 input) { return sint16x8{ _mm_cvtepu8_epi16(input.data) }; }
	template<> sint32x4 expand_low<sint32x4, uint16x8>(uint16x8 input) { return sint32x4{ _mm_cvtepu16_epi32(input.data) }; }
	template<> sint64x2 expand_low<sint64x2, uint32x4>(uint32x4 input) { return sint64x2{ _mm_cvtepu32_epi64(input.data) }; }
	template<> sint16x8 expand_low<sint16x8, sint8x16>(sint8x16 input) { return sint16x8{ _mm_cvtepi8_epi16(input.data) }; }
	template<> sint32x4 expand_low<sint32x4, sint16x8>(sint16x8 input) { return sint32x4{ _mm_cvtepi16_epi32(input.data) }; }
	template<> sint64x2 expand_low<sint64x2, sint32x4>(sint32x4 input) { return sint64x2{ _mm_cvtepi32_epi64(input.data) }; }
	template<> uint16x8 expand_low<uint16x8, sint8x16>(sint8x16 input) { return uint16x8{ _mm_cvtepi8_epi16(input.data) }; }
	template<> uint32x4 expand_low<uint32x4, sint16x8>(sint16x8 input) { return uint32x4{ _mm_cvtepi16_epi32(input.data) }; }
	template<> uint64x2 expand_low<uint64x2, sint32x4>(sint32x4 input) { return uint64x2{ _mm_cvtepi32_epi64(input.data) }; }

	template<> uint16x16 expand_low<uint16x16, uint8x32>(uint8x32 input) { return uint16x16{ _mm256_cvtepu8_epi16(detail::split_low(input.data)) }; }
	template<> uint32x8 expand_low<uint32x8, uint16x16>(uint16x16 input) { return uint32x8{ _mm256_cvtepu16_epi32(detail::split_low(input.data)) }; }
	template<> uint64x4 expand_low<uint64x4, uint32x8>(uint32x8 input) { return uint64x4{ _mm256_cvtepu32_epi64(detail::split_low(input.data)) }; }
	template<> sint16x16 expand_low<sint16x16, uint8x32>(uint8x32 input) { return sint16x16{ _mm256_cvtepu8_epi16(detail::split_low(input.data)) }; }
	template<> sint32x8 expand_low<sint32x8, uint16x16>(uint16x16 input) { return sint32x8{ _mm256_cvtepu16_epi32(detail::split_low(input.data)) }; }
	template<> sint64x4 expand_low<sint64x4, uint32x8>(uint32x8 input) { return sint64x4{ _mm256_cvtepu32_epi64(detail::split_low(input.data)) }; }
	template<> sint16x16 expand_low<sint16x16, sint8x32>(sint8x32 input) { return sint16x16{ _mm256_cvtepi8_epi16(detail::split_low(input.data)) }; }
	template<> sint32x8 expand_low<sint32x8, sint16x16>(sint16x16 input) { return sint32x8{ _mm256_cvtepi16_epi32(detail::split_low(input.data)) }; }
	template<> sint64x4 expand_low<sint64x4, sint32x8>(sint32x8 input) { return sint64x4{ _mm256_cvtepi32_epi64(detail::split_low(input.data)) }; }
	template<> uint16x16 expand_low<uint16x16, sint8x32>(sint8x32 input) { return uint16x16{ _mm256_cvtepi8_epi16(detail::split_low(input.data)) }; }
	template<> uint32x8 expand_low<uint32x8, sint16x16>(sint16x16 input) { return uint32x8{ _mm256_cvtepi16_epi32(detail::split_low(input.data)) }; }
	template<> uint64x4 expand_low<uint64x4, sint32x8>(sint32x8 input) { return uint64x4{ _mm256_cvtepi32_epi64(detail::split_low(input.data)) }; }
	template<> float64x2 expand_low<float64x2, float32x4>(float32x4 input) { return float64x2{ _mm_cvtps_pd(input.data) }; }
	template<> float64x4 expand_low<float64x4, float32x8>(float32x8 input) { return float64x4{ _mm256_cvtps_pd(detail::split_low(input.data)) }; }
#if defined(_FOYE_SIMD_HAS_FP16_)
	template<> float32x4 expand_low<float32x4, float16x8>(float16x8 input) { return float32x4{ detail::split_low(_mm256_cvtph_ps(input.data)) }; }
	template<> float32x8 expand_low<float32x8, float16x16>(float16x16 input) { return float32x8{ _mm256_cvtph_ps(detail::split_low(input.data)) }; }
#endif
#if defined(_FOYE_SIMD_HAS_BF16_)
	template<> float32x4 expand_low<float32x4, bfloat16x8>(bfloat16x8 input) 
	{
		return float32x4{ detail::split_low(
			_mm256_castsi256_ps(
				_mm256_slli_epi32(
					_mm256_cvtepu16_epi32(input.data), 16))) };
	}

	template<> float32x8 expand_low<float32x8, bfloat16x16>(bfloat16x16 input) 
	{
		return float32x8{ _mm256_castsi256_ps(
			_mm256_slli_epi32(
				_mm256_cvtepu16_epi32(
					detail::split_low(input.data)), 16)) };
	}
#endif

	template<> uint16x8 expand_high<uint16x8, uint8x16>(uint8x16 input) { return uint16x8{ _mm_cvtepu8_epi16(_mm_srli_si128(input.data, 8)) }; }
	template<> uint32x4 expand_high<uint32x4, uint16x8>(uint16x8 input) { return uint32x4{ _mm_cvtepu16_epi32(_mm_srli_si128(input.data, 8)) }; }
	template<> uint64x2 expand_high<uint64x2, uint32x4>(uint32x4 input) { return uint64x2{ _mm_cvtepu32_epi64(_mm_srli_si128(input.data, 8)) }; }
	template<> sint16x8 expand_high<sint16x8, uint8x16>(uint8x16 input) { return sint16x8{ _mm_cvtepu8_epi16(_mm_srli_si128(input.data, 8)) }; }
	template<> sint32x4 expand_high<sint32x4, uint16x8>(uint16x8 input) { return sint32x4{ _mm_cvtepu16_epi32(_mm_srli_si128(input.data, 8)) }; }
	template<> sint64x2 expand_high<sint64x2, uint32x4>(uint32x4 input) { return sint64x2{ _mm_cvtepu32_epi64(_mm_srli_si128(input.data, 8)) }; }
	template<> sint16x8 expand_high<sint16x8, sint8x16>(sint8x16 input) { return sint16x8{ _mm_cvtepi8_epi16(_mm_srli_si128(input.data, 8)) }; }
	template<> sint32x4 expand_high<sint32x4, sint16x8>(sint16x8 input) { return sint32x4{ _mm_cvtepi16_epi32(_mm_srli_si128(input.data, 8)) }; }
	template<> sint64x2 expand_high<sint64x2, sint32x4>(sint32x4 input) { return sint64x2{ _mm_cvtepi32_epi64(_mm_srli_si128(input.data, 8)) }; }
	template<> uint16x8 expand_high<uint16x8, sint8x16>(sint8x16 input) { return uint16x8{ _mm_cvtepi8_epi16(_mm_srli_si128(input.data, 8)) }; }
	template<> uint32x4 expand_high<uint32x4, sint16x8>(sint16x8 input) { return uint32x4{ _mm_cvtepi16_epi32(_mm_srli_si128(input.data, 8)) }; }
	template<> uint64x2 expand_high<uint64x2, sint32x4>(sint32x4 input) { return uint64x2{ _mm_cvtepi32_epi64(_mm_srli_si128(input.data, 8)) }; }
	template<> uint16x16 expand_high<uint16x16, uint8x32>(uint8x32 input) { return uint16x16{ _mm256_cvtepu8_epi16(detail::split_high(input.data)) }; }
	template<> uint32x8 expand_high<uint32x8, uint16x16>(uint16x16 input) { return uint32x8{ _mm256_cvtepu16_epi32(detail::split_high(input.data)) }; }
	template<> uint64x4 expand_high<uint64x4, uint32x8>(uint32x8 input) { return uint64x4{ _mm256_cvtepu32_epi64(detail::split_high(input.data)) }; }
	template<> sint16x16 expand_high<sint16x16, uint8x32>(uint8x32 input) { return sint16x16{ _mm256_cvtepu8_epi16(detail::split_high(input.data)) }; }
	template<> sint32x8 expand_high<sint32x8, uint16x16>(uint16x16 input) { return sint32x8{ _mm256_cvtepu16_epi32(detail::split_high(input.data)) }; }
	template<> sint64x4 expand_high<sint64x4, uint32x8>(uint32x8 input) { return sint64x4{ _mm256_cvtepu32_epi64(detail::split_high(input.data)) }; }
	template<> sint16x16 expand_high<sint16x16, sint8x32>(sint8x32 input) { return sint16x16{ _mm256_cvtepi8_epi16(detail::split_high(input.data)) }; }
	template<> sint32x8 expand_high<sint32x8, sint16x16>(sint16x16 input) { return sint32x8{ _mm256_cvtepi16_epi32(detail::split_high(input.data)) }; }
	template<> sint64x4 expand_high<sint64x4, sint32x8>(sint32x8 input) { return sint64x4{ _mm256_cvtepi32_epi64(detail::split_high(input.data)) }; }
	template<> uint16x16 expand_high<uint16x16, sint8x32>(sint8x32 input) { return uint16x16{ _mm256_cvtepi8_epi16(detail::split_high(input.data)) }; }
	template<> uint32x8 expand_high<uint32x8, sint16x16>(sint16x16 input) { return uint32x8{ _mm256_cvtepi16_epi32(detail::split_high(input.data)) }; }
	template<> uint64x4 expand_high<uint64x4, sint32x8>(sint32x8 input) { return uint64x4{ _mm256_cvtepi32_epi64(detail::split_high(input.data)) }; }
	template<> float64x2 expand_high<float64x2, float32x4>(float32x4 input) { return float64x2{ _mm_cvtps_pd(_mm_movehl_ps(input.data, input.data)) }; }
	template<> float64x4 expand_high<float64x4, float32x8>(float32x8 input) { return float64x4{ _mm256_cvtps_pd(detail::split_high(input.data)) }; }
#if defined(_FOYE_SIMD_HAS_FP16_)
	template<> float32x4 expand_high<float32x4, float16x8>(float16x8 input) { return float32x4{ detail::split_high(_mm256_cvtph_ps(input.data)) }; }
	template<> float32x8 expand_high<float32x8, float16x16>(float16x16 input) { return float32x8{ _mm256_cvtph_ps(detail::split_high(input.data)) }; }
#endif
#if defined(_FOYE_SIMD_HAS_BF16_)
	template<> float32x4 expand_high<float32x4, bfloat16x8>(bfloat16x8 input) 
	{
		return float32x4{ detail::split_high(
			_mm256_castsi256_ps(_mm256_slli_epi32(
				_mm256_cvtepu16_epi32(input.data), 16))) };
	}

	template<> float32x8 expand_high<float32x8, bfloat16x16>(bfloat16x16 input) 
	{
		return float32x8{ _mm256_castsi256_ps(
			_mm256_slli_epi32(_mm256_cvtepu16_epi32(
				detail::split_high(input.data)), 16)) };
	}
#endif

	template<> uint32x4 narrowing(uint64x4 source)
	{
		__m256i mask = _mm256_set_epi32(0, 0, 0, 0, 6, 4, 2, 0);
		__m256i permuted = _mm256_permutevar8x32_epi32(source.data, mask);
		__m128i result = _mm256_castsi256_si128(permuted);
		return uint32x4{ result };
	}

	template<> uint16x8 narrowing(uint32x8 source)
	{
		__m256i shifted = _mm256_srli_epi32(source.data, 16);
		__m256i masked = _mm256_and_si256(source.data, _mm256_set1_epi32(0xFFFF));
		__m128i low_mask = _mm256_extractf128_si256(masked, 0);
		__m128i high_mask = _mm256_extractf128_si256(masked, 1);
		return uint16x8{ _mm_packus_epi32(low_mask, high_mask) };
	}

	template<> sint8x16 narrowing(uint16x16 source)
	{
		const __m256i mask = _mm256_set1_epi16(0x00FF);
		__m256i truncated_16bit = _mm256_and_si256(source, mask);
		__m128i low = _mm256_extractf128_si256(truncated_16bit, 0);
		__m128i high = _mm256_extractf128_si256(truncated_16bit, 1);
		return sint8x16{ _mm_packus_epi16(low, high) };
	}

	template<> sint32x4 narrowing(uint64x4 source) { return fyx::simd::reinterpret<sint32x4>(fyx::simd::narrowing<uint32x4>(source)); }
	template<> uint32x4 narrowing(sint64x4 source) { return fyx::simd::narrowing<uint32x4>(fyx::simd::reinterpret<uint64x4>(source)); }
	template<> sint32x4 narrowing(sint64x4 source) { return fyx::simd::reinterpret<sint32x4>(fyx::simd::narrowing<uint32x4>(source)); }
	template<> sint16x8 narrowing(uint32x8 source) { return fyx::simd::reinterpret<sint16x8>(fyx::simd::narrowing<uint16x8>(source)); }
	template<> sint16x8 narrowing(sint32x8 source) { return fyx::simd::narrowing<sint16x8>(fyx::simd::reinterpret<uint32x8>(source)); }
	template<> uint16x8 narrowing(sint32x8 source) { return fyx::simd::narrowing<uint16x8>(fyx::simd::reinterpret<uint32x8>(source)); }
	template<> uint8x16 narrowing(uint16x16 source) { return fyx::simd::reinterpret<uint8x16>(fyx::simd::narrowing<sint8x16>(source)); }
	template<> sint8x16 narrowing(sint16x16 source) { return fyx::simd::narrowing<sint8x16>(fyx::simd::reinterpret<uint16x16>(source)); }
	template<> uint8x16 narrowing(sint16x16 source) { return fyx::simd::reinterpret<uint8x16>(fyx::simd::narrowing<sint8x16>(source)); }

	template<> float32x4 narrowing(float64x4 source) { return float32x4{ _mm256_cvtpd_ps(source.data) }; }
	
	template<> sint16x8 narrowing(float32x8 source) { return fyx::simd::narrowing<sint16x8>(fyx::simd::trunc_as_i(source)); }
	template<> uint16x8 narrowing(float32x8 source) { return fyx::simd::narrowing<uint16x8>(fyx::simd::trunc_as_i(source)); }
	
#if defined(_FOYE_SIMD_HAS_FP16_)
	template<> float16x8 narrowing(float32x8 source) 
	{
		return float16x8{ _mm256_cvtps_ph(source.data, _MM_FROUND_CUR_DIRECTION) }; 
	}
#endif
#if defined(_FOYE_SIMD_HAS_BF16_)
	template<> bfloat16x8 narrowing(float32x8 source) 
	{
		const __m256i v_exp_mask = _mm256_set1_epi32(0x7F800000);
		const __m256i v_mant_mask = _mm256_set1_epi32(0x007FFFFF);
		const __m256i v_zero = _mm256_setzero_si256();

		const __m256i v = _mm256_castps_si256(source.data);

		const __m256i is_nan = _mm256_and_si256(
			_mm256_cmpeq_epi32(_mm256_and_si256(v, v_exp_mask), v_exp_mask),
			_mm256_cmpgt_epi32(_mm256_and_si256(v, v_mant_mask), v_zero));

		__m256i shifted = _mm256_srli_epi32(v, 16);
		__m256i shifted_low = _mm256_and_si256(shifted, _mm256_set1_epi32(0x0000FFFF));

		__m256i shifted_updata = _mm256_or_si256(shifted,
			_mm256_and_si256(
				_mm256_set1_epi32(0x00000001),
				_mm256_and_si256(is_nan, _mm256_cmpeq_epi32(shifted_low, v_zero))));

		return bfloat16x8{ _mm_packus_epi32(
			_mm256_extractf128_si256(shifted_updata, 0),
			_mm256_extractf128_si256(shifted_updata, 1)) };
	}
#endif

	template<> float32x8 floating(sint16x8 input) { return float32x8{ _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(input.data)) }; }
	template<> float32x8 floating(uint16x8 input) { return float32x8{ _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(input.data)) }; }
	template<> float32x8 floating(sint32x8 input) { return float32x8{ _mm256_cvtepi32_ps(input.data) }; }
	template<> float64x4 floating(sint32x4 input) { return float64x4{ _mm256_cvtepi32_pd(input.data) }; }
	template<> float32x4 floating(sint32x4 input) { return float32x4{ _mm_cvtepi32_ps(input.data) }; }
	template<> float32x4 floating(uint32x4 input) { return float32x4{ _mm_cvtepu32_ps(input.data) }; }
	template<> float64x4 floating(uint32x4 input) { return float64x4{ _mm256_cvtps_pd(_mm_cvtepu32_ps(input.data)) }; }

	template<> float32x8 floating(uint32x8 input)
	{
		__m256 result = detail::merge(
			_mm_cvtepu32_ps(detail::split_high(input.data)),
			_mm_cvtepu32_ps(detail::split_low(input.data)));
		return float32x8{ result };
	}

	template<> float64x2 floating(uint64x2 input)
	{
		__m128i v = input.data;
		const __m128i magic_i_hi32 = _mm_set1_epi64x(0x4530000080000000);
		const __m128i magic_i_all = _mm_set1_epi64x(0x4530000080100000);
		const __m128d magic_d_all = _mm_castsi128_pd(magic_i_all);
		const __m128i magic_i_lo = _mm_set1_epi64x(0x4330000000000000);

		__m128i v_lo = _mm_blend_epi16(v, magic_i_lo, 0xcc);
		__m128i v_hi = _mm_srli_epi64(v, 32);

		v_hi = _mm_xor_si128(v_hi, magic_i_hi32);

		__m128d v_hi_dbl = _mm_sub_pd(_mm_castsi128_pd(v_hi), magic_d_all);
		__m128d result = _mm_add_pd(v_hi_dbl, _mm_castsi128_pd(v_lo));
		return float64x2{ result };
	}

	template<> float64x4 floating(uint64x4 input)
	{
		__m256i v = input.data;
		const __m256i magic_i_lo = _mm256_set1_epi64x(0x4330000000000000);
		const __m256i magic_i_hi32 = _mm256_set1_epi64x(0x4530000080000000);
		const __m256i magic_i_all = _mm256_set1_epi64x(0x4530000080100000);
		const __m256d magic_d_all = _mm256_castsi256_pd(magic_i_all);

		__m256i v_lo = _mm256_blend_epi32(magic_i_lo, v, 0x55);
		__m256i v_hi = _mm256_srli_epi64(v, 32);

		v_hi = _mm256_xor_si256(v_hi, magic_i_hi32);

		__m256d v_hi_dbl = _mm256_sub_pd(_mm256_castsi256_pd(v_hi), magic_d_all);
		__m256d result = _mm256_add_pd(v_hi_dbl, _mm256_castsi256_pd(v_lo));
		return float64x4{ result };
	}

	template<> float64x2 floating(sint64x2 input)
	{
		__m128i sign_mask = _mm_srai_epi64(input.data, 63);
		__m128i unsigned_input = _mm_add_epi64(
			_mm_xor_si128(input.data, sign_mask),
			_mm_srli_epi64(sign_mask, 63)
		);

		float64x2 result = fyx::simd::floating<float64x2, uint64x2>(
			uint64x2{ unsigned_input });

		__m128i sign_bits = _mm_slli_epi64(sign_mask, 63);
		result.data = _mm_xor_pd(result.data, _mm_castsi128_pd(sign_bits));

		return result;
	}

	template<> float64x4 floating(sint64x4 input)
	{
		__m256i sign_mask = _mm256_srai_epi64(input.data, 63);
		__m256i unsigned_input = _mm256_add_epi64(
			_mm256_xor_si256(input.data, sign_mask),
			_mm256_srli_epi64(sign_mask, 63)
		);

		float64x4 result = fyx::simd::floating<float64x4, uint64x4>(
			uint64x4{ unsigned_input });

		__m256i sign_bits = _mm256_slli_epi64(sign_mask, 63);
		result.data = _mm256_xor_pd(result.data, _mm256_castsi256_pd(sign_bits));
		return result;
	}

#if defined(_FOYE_SIMD_HAS_FP16_) || defined(_FOYE_SIMD_HAS_BF16_)
#define DEFINE_AS_HALF(dst_vtype, cvtfunc) \
template<> dst_vtype##x8 floating(uint32x8 input)\
{\
	__m128 low_f = _mm_cvtepu32_ps(detail::split_low(input.data));\
	__m128 high_f = _mm_cvtepu32_ps(detail::split_high(input.data));\
	return dst_vtype##x8{ cvtfunc(detail::merge(high_f, low_f)) };\
}\
template<> dst_vtype##x8 floating(sint16x8 input)\
{\
	__m256i vi32 = _mm256_cvtepi16_epi32(input.data);\
	__m256 vf32 = _mm256_cvtepi32_ps(vi32);\
	return dst_vtype##x8{ cvtfunc(vf32) };\
}\
template<> dst_vtype##x8 floating(uint16x8 input) \
{ \
    return dst_vtype##x8{ cvtfunc( \
        _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(input.data)) \
    ) }; \
} \
template<> dst_vtype##x8 floating(sint32x8 input)\
{\
	return dst_vtype##x8{ cvtfunc(_mm256_cvtepi32_ps(input.data)) };\
}\
template<> dst_vtype##x16 floating(uint16x16 input)\
{\
	__m256 vf32_low = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32((detail::split_low(input.data))));\
	__m256 vf32_high = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32((detail::split_high(input.data))));\
	return dst_vtype##x16{ detail::merge(\
		cvtfunc(vf32_low),\
		cvtfunc(vf32_high)) };\
}\
template<> dst_vtype##x16 floating(uint8x16 input)\
{\
	__m256 low_f32 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(input.data));\
	__m256 high_f32 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_unpackhi_epi64(input.data, input.data)));\
	return dst_vtype##x16{ detail::merge(\
		cvtfunc(low_f32),\
		cvtfunc(high_f32)) };\
}\
template<> dst_vtype##x16 floating(sint16x16 input)\
{\
	__m256 vf32_low = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32((detail::split_low(input.data))));\
	__m256 vf32_high = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32((detail::split_high(input.data))));\
	return dst_vtype##x16{ detail::merge(\
		cvtfunc(vf32_low),\
		cvtfunc(vf32_high)) };\
}\
template<> dst_vtype##x16 floating(sint8x16 input)\
{\
	__m256 low_f32 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(input.data));\
	__m256 high_f32 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_unpackhi_epi64(input.data, input.data)));\
	return dst_vtype##x16{ detail::merge(\
		cvtfunc(low_f32),\
		cvtfunc(high_f32)) };\
}
#else
#define DEFINE_AS_HALF(...)
#endif
#if defined(_FOYE_SIMD_HAS_FP16_)
	DEFINE_AS_HALF(float16, cvt8lane_fp32_to_fp16)
#endif

#if defined(_FOYE_SIMD_HAS_BF16_)
	DEFINE_AS_HALF(bfloat16, cvt8lane_fp32_to_bf16)
#endif
}

#if 0
namespace fyx::simd
{
	template<typename target_simd_type, typename source_simd_type>
	target_simd_type narrowing_sat(source_simd_type source) { __assume(false); }

	template<> uint8x16 narrowing_sat<uint8x16, uint16x16>(uint16x16 source);
	template<> uint8x16 narrowing_sat<uint8x16, sint16x16>(sint16x16 source);
	template<> uint8x16 narrowing_sat<uint8x16, sint8x16>(sint8x16 source);

	template<> sint8x16 narrowing_sat<sint8x16, uint16x16>(uint16x16 source);
	template<> sint8x16 narrowing_sat<sint8x16, sint16x16>(sint16x16 source);
	template<> sint8x16 narrowing_sat<sint8x16, uint8x16>(uint8x16 source);

	template<> uint16x8 narrowing_sat<uint16x8, sint32x8>(sint32x8 source);
	template<> uint16x8 narrowing_sat<uint16x8, sint32x8>(sint32x8 source);
	template<> uint16x8 narrowing_sat<uint16x8, sint16x8>(sint16x8 source);

	template<> sint16x8 narrowing_sat<sint16x8, sint32x8>(sint32x8 source);
	template<> sint16x8 narrowing_sat<sint16x8, sint32x8>(sint32x8 source);
	template<> sint16x8 narrowing_sat<sint16x8, uint16x8>(uint16x8 source);

	template<> uint32x4 narrowing_sat<uint32x4, sint64x4>(sint64x4 source);
	template<> uint32x4 narrowing_sat<uint32x4, sint64x4>(sint64x4 source);
	template<> uint32x4 narrowing_sat<uint32x4, sint32x4>(sint32x4 source);

	template<> sint32x4 narrowing_sat<sint32x4, sint64x4>(sint64x4 source);
	template<> sint32x4 narrowing_sat<sint32x4, sint64x4>(sint64x4 source);
	template<> sint32x4 narrowing_sat<sint32x4, uint32x4>(uint32x4 source);

	template<> sint64x4 narrowing_sat<sint64x4, uint64x4>(uint64x4 source);
	template<> uint64x4 narrowing_sat<uint64x4, sint64x4>(sint64x4 source);

#if defined(_FOYE_SIMD_HAS_FP16_)
	template<> float16x8 narrowing_sat<float16x8, sint16x8>(sint16x8 source);
	template<> float16x8 narrowing_sat<float16x8, uint16x8>(uint16x8 source);
	template<> float16x8 narrowing_sat<float16x8, sint32x8>(sint32x8 source);
	template<> float16x8 narrowing_sat<float16x8, uint32x8>(uint32x8 source);
	template<> float16x16 narrowing_sat<float16x16, sint16x16>(sint16x16 source);
	template<> float16x16 narrowing_sat<float16x16, uint16x16>(uint16x16 source);
#endif

#if defined(_FOYE_SIMD_HAS_BF16_)
	template<> bfloat16x8 narrowing_sat<bfloat16x8, sint16x8>(sint16x8 source);
	template<> bfloat16x8 narrowing_sat<bfloat16x8, uint16x8>(uint16x8 source);
	template<> bfloat16x8 narrowing_sat<bfloat16x8, sint32x8>(sint32x8 source);
	template<> bfloat16x8 narrowing_sat<bfloat16x8, uint32x8>(uint32x8 source);
	template<> bfloat16x16 narrowing_sat<bfloat16x16, sint16x16>(sint16x16 source);
	template<> bfloat16x16 narrowing_sat<bfloat16x16, uint16x16>(uint16x16 source);
#endif
}
#endif






namespace fyx::simd
{
	template<typename target_simd_type, typename source_simd_type>
	target_simd_type convert_std(source_simd_type source)
	{
		static_assert(target_simd_type::lane_width == source_simd_type::lane_width,
			"simd type conversions require the same lane width => target_simd_type::lane_width == source_simd_type::lane_width");

		using target_scalar_t = typename target_simd_type::scalar_t;
		using source_scalar_t = typename source_simd_type::scalar_t;

		if constexpr (std::is_integral_v<source_scalar_t> 
				   && std::is_integral_v<target_scalar_t>
				   && (sizeof(source_scalar_t) == sizeof(target_scalar_t)))
		{
			return fyx::simd::reinterpret<target_simd_type, source_simd_type>(source);
		}

		if constexpr (std::is_integral_v<source_scalar_t> 
				   && fyx::simd::is_floating_basic_simd_v<target_simd_type>)
		{
			return fyx::simd::floating<target_simd_type, source_simd_type>(source);
		}

		if constexpr (fyx::simd::is_floating_basic_simd_v<source_simd_type>
				   && fyx::simd::is_integral_basic_simd_v<target_simd_type>)
		{
			return fyx::simd::reinterpret<target_simd_type>(
				fyx::simd::trunc_as_i(source));
		}

		if constexpr (fyx::simd::is_floating_basic_simd_v<source_simd_type> && fyx::simd::is_floating_basic_simd_v<target_simd_type>
			&& fyx::simd::is_integral_basic_simd_v<source_simd_type> && fyx::simd::is_integral_basic_simd_v<target_simd_type>)
		{
			if constexpr (sizeof(target_scalar_t) > sizeof(source_scalar_t))
			{
				return fyx::simd::expand<target_simd_type, source_simd_type>(source);
			}
			else
			{
				return fyx::simd::narrowing<target_simd_type, source_simd_type>(source);
			}
		}
		else
		{
			__assume(false);
		}
	}


}



#endif
