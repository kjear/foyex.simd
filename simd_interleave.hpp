#ifndef _FOYE_SIMD_INTERLEAVE_HPP_
#define _FOYE_SIMD_INTERLEAVE_HPP_
#pragma once

#include "simd_def.hpp"
#include "simd_cvt.hpp"

namespace fyx::simd::detail
{
    enum class interleave_store_mode { aligned, unaligned, stream };
    enum class interleave_load_mode { aligned, unaligned };

    template<typename simd_type, fyx::simd::detail::interleave_load_mode mode> requires((sizeof(typename simd_type::scalar_t) == sizeof(std::uint8_t))
        && fyx::simd::is_128bits_simd_v<simd_type>)
        void load_interleave_impl_lane2(const void* ptr, simd_type* a, simd_type* b)
    {
        __m128i t00, t01;
        if constexpr (mode == fyx::simd::detail::interleave_load_mode::aligned)
        {
            t00 = _mm_load_si128(reinterpret_cast<const __m128i*>(reinterpret_cast<const simd_type::scalar_t*>(ptr) + (simd_type::lane_width * 0)));
            t01 = _mm_load_si128(reinterpret_cast<const __m128i*>(reinterpret_cast<const simd_type::scalar_t*>(ptr) + (simd_type::lane_width * 1)));
        }
        else
        {
            t00 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(reinterpret_cast<const simd_type::scalar_t*>(ptr) + (simd_type::lane_width * 0)));
            t01 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(reinterpret_cast<const simd_type::scalar_t*>(ptr) + (simd_type::lane_width * 1)));
        }

        __m128i t10 = _mm_unpacklo_epi8(t00, t01);
        __m128i t11 = _mm_unpackhi_epi8(t00, t01);

        __m128i t20 = _mm_unpacklo_epi8(t10, t11);
        __m128i t21 = _mm_unpackhi_epi8(t10, t11);

        __m128i t30 = _mm_unpacklo_epi8(t20, t21);
        __m128i t31 = _mm_unpackhi_epi8(t20, t21);

        (*a).data = _mm_unpacklo_epi8(t30, t31);
        (*b).data = _mm_unpackhi_epi8(t30, t31);
    }

    template<typename simd_type, fyx::simd::detail::interleave_load_mode mode> requires((sizeof(typename simd_type::scalar_t) == sizeof(std::uint8_t))
        && fyx::simd::is_128bits_simd_v<simd_type>)
        void load_interleave_impl_lane3(const void* ptr, simd_type* a, simd_type* b, simd_type* c)
    {
        __m128i s0, s1, s2;
        if constexpr (mode == fyx::simd::detail::interleave_load_mode::aligned)
        {
            s0 = _mm_load_si128(reinterpret_cast<const __m128i*>(reinterpret_cast<const simd_type::scalar_t*>(ptr) + (simd_type::lane_width * 0)));
            s1 = _mm_load_si128(reinterpret_cast<const __m128i*>(reinterpret_cast<const simd_type::scalar_t*>(ptr) + (simd_type::lane_width * 1)));
            s2 = _mm_load_si128(reinterpret_cast<const __m128i*>(reinterpret_cast<const simd_type::scalar_t*>(ptr) + (simd_type::lane_width * 2)));
        }
        else
        {
            s0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(reinterpret_cast<const simd_type::scalar_t*>(ptr) + (simd_type::lane_width * 0)));
            s1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(reinterpret_cast<const simd_type::scalar_t*>(ptr) + (simd_type::lane_width * 1)));
            s2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(reinterpret_cast<const simd_type::scalar_t*>(ptr) + (simd_type::lane_width * 2)));
        }

        const __m128i m0 = _mm_setr_epi8(0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0);
        const __m128i m1 = _mm_setr_epi8(0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0);

        __m128i a0 = _mm_blendv_epi8(_mm_blendv_epi8(s0, s1, m0), s2, m1);
        __m128i b0 = _mm_blendv_epi8(_mm_blendv_epi8(s1, s2, m0), s0, m1);
        __m128i c0 = _mm_blendv_epi8(_mm_blendv_epi8(s2, s0, m0), s1, m1);

        const __m128i sh_b = _mm_setr_epi8(0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14, 1, 4, 7, 10, 13);
        const __m128i sh_g = _mm_setr_epi8(1, 4, 7, 10, 13, 0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14);
        const __m128i sh_r = _mm_setr_epi8(2, 5, 8, 11, 14, 1, 4, 7, 10, 13, 0, 3, 6, 9, 12, 15);

        (*a).data = _mm_shuffle_epi8(a0, sh_b);
        (*b).data = _mm_shuffle_epi8(b0, sh_g);
        (*c).data = _mm_shuffle_epi8(c0, sh_r);
    }

    template<typename simd_type, fyx::simd::detail::interleave_load_mode mode> requires((sizeof(typename simd_type::scalar_t) == sizeof(std::uint8_t))
        && fyx::simd::is_128bits_simd_v<simd_type>)
        void load_interleave_impl_lane4(const void* ptr, simd_type* a, simd_type* b, simd_type* c, simd_type* d)
    {
        __m128i u0, u1, u2, u3;
        if constexpr (mode == fyx::simd::detail::interleave_load_mode::aligned)
        {
            u0 = _mm_load_si128(reinterpret_cast<const __m128i*>(reinterpret_cast<const simd_type::scalar_t*>(ptr) + (simd_type::lane_width * 0)));
            u1 = _mm_load_si128(reinterpret_cast<const __m128i*>(reinterpret_cast<const simd_type::scalar_t*>(ptr) + (simd_type::lane_width * 1)));
            u2 = _mm_load_si128(reinterpret_cast<const __m128i*>(reinterpret_cast<const simd_type::scalar_t*>(ptr) + (simd_type::lane_width * 2)));
            u3 = _mm_load_si128(reinterpret_cast<const __m128i*>(reinterpret_cast<const simd_type::scalar_t*>(ptr) + (simd_type::lane_width * 3)));
        }
        else
        {
            u0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(reinterpret_cast<const simd_type::scalar_t*>(ptr) + (simd_type::lane_width * 0)));
            u1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(reinterpret_cast<const simd_type::scalar_t*>(ptr) + (simd_type::lane_width * 1)));
            u2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(reinterpret_cast<const simd_type::scalar_t*>(ptr) + (simd_type::lane_width * 2)));
            u3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(reinterpret_cast<const simd_type::scalar_t*>(ptr) + (simd_type::lane_width * 3)));
        }

        __m128i v0 = _mm_unpacklo_epi8(u0, u2);
        __m128i v1 = _mm_unpackhi_epi8(u0, u2);
        __m128i v2 = _mm_unpacklo_epi8(u1, u3);
        __m128i v3 = _mm_unpackhi_epi8(u1, u3);

        u0 = _mm_unpacklo_epi8(v0, v2);
        u1 = _mm_unpacklo_epi8(v1, v3);
        u2 = _mm_unpackhi_epi8(v0, v2);
        u3 = _mm_unpackhi_epi8(v1, v3);

        v0 = _mm_unpacklo_epi8(u0, u1);
        v1 = _mm_unpacklo_epi8(u2, u3);
        v2 = _mm_unpackhi_epi8(u0, u1);
        v3 = _mm_unpackhi_epi8(u2, u3);

        (*a).data = _mm_unpacklo_epi8(v0, v1);
        (*b).data = _mm_unpackhi_epi8(v0, v1);
        (*c).data = _mm_unpacklo_epi8(v2, v3);
        (*d).data = _mm_unpackhi_epi8(v2, v3);
    }


    template<typename simd_type, fyx::simd::detail::interleave_load_mode mode>
        requires((sizeof(typename simd_type::scalar_t) == sizeof(std::uint16_t))
    && fyx::simd::is_128bits_simd_v<simd_type>)
        void load_interleave_impl_lane2(const void* ptr, simd_type* a, simd_type* b)
    {
        __m128i v0, v1;
        if constexpr (mode == fyx::simd::detail::interleave_load_mode::aligned)
        {
            v0 = _mm_load_si128(reinterpret_cast<const __m128i*>(reinterpret_cast<const simd_type::scalar_t*>(ptr) + (simd_type::lane_width * 0)));
            v1 = _mm_load_si128(reinterpret_cast<const __m128i*>(reinterpret_cast<const simd_type::scalar_t*>(ptr) + (simd_type::lane_width * 1)));
        }
        else
        {
            v0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(reinterpret_cast<const simd_type::scalar_t*>(ptr) + (simd_type::lane_width * 0)));
            v1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(reinterpret_cast<const simd_type::scalar_t*>(ptr) + (simd_type::lane_width * 1)));
        }

        __m128i v2 = _mm_unpacklo_epi16(v0, v1);
        __m128i v3 = _mm_unpackhi_epi16(v0, v1);
        __m128i v4 = _mm_unpacklo_epi16(v2, v3);
        __m128i v5 = _mm_unpackhi_epi16(v2, v3);

        (*a).data = _mm_unpacklo_epi16(v4, v5);
        (*b).data = _mm_unpackhi_epi16(v4, v5);
    }

    template<typename simd_type, fyx::simd::detail::interleave_load_mode mode>
        requires((sizeof(typename simd_type::scalar_t) == sizeof(std::uint16_t))
    && fyx::simd::is_128bits_simd_v<simd_type>)
        void load_interleave_impl_lane3(const void* ptr, simd_type* a, simd_type* b, simd_type* c)
    {
        __m128i v0, v1, v2;
        if constexpr (mode == fyx::simd::detail::interleave_load_mode::aligned)
        {
            v0 = _mm_load_si128(reinterpret_cast<const __m128i*>(reinterpret_cast<const simd_type::scalar_t*>(ptr) + (simd_type::lane_width * 0)));
            v1 = _mm_load_si128(reinterpret_cast<const __m128i*>(reinterpret_cast<const simd_type::scalar_t*>(ptr) + (simd_type::lane_width * 1)));
            v2 = _mm_load_si128(reinterpret_cast<const __m128i*>(reinterpret_cast<const simd_type::scalar_t*>(ptr) + (simd_type::lane_width * 2)));
        }
        else
        {
            v0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(reinterpret_cast<const simd_type::scalar_t*>(ptr) + (simd_type::lane_width * 0)));
            v1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(reinterpret_cast<const simd_type::scalar_t*>(ptr) + (simd_type::lane_width * 1)));
            v2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(reinterpret_cast<const simd_type::scalar_t*>(ptr) + (simd_type::lane_width * 2)));
        }

        __m128i a0 = _mm_blend_epi16(_mm_blend_epi16(v0, v1, 0x92), v2, 0x24);
        __m128i b0 = _mm_blend_epi16(_mm_blend_epi16(v2, v0, 0x92), v1, 0x24);
        __m128i c0 = _mm_blend_epi16(_mm_blend_epi16(v1, v2, 0x92), v0, 0x24);

        const __m128i sh_a = _mm_setr_epi8(0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15, 4, 5, 10, 11);
        const __m128i sh_b = _mm_setr_epi8(2, 3, 8, 9, 14, 15, 4, 5, 10, 11, 0, 1, 6, 7, 12, 13);
        const __m128i sh_c = _mm_setr_epi8(4, 5, 10, 11, 0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15);

        (*a).data = _mm_shuffle_epi8(a0, sh_a);
        (*b).data = _mm_shuffle_epi8(b0, sh_b);
        (*c).data = _mm_shuffle_epi8(c0, sh_c);
    }

    template<typename simd_type, fyx::simd::detail::interleave_load_mode mode>
        requires((sizeof(typename simd_type::scalar_t) == sizeof(std::uint16_t))
    && fyx::simd::is_128bits_simd_v<simd_type>)
        void load_interleave_impl_lane4(const void* ptr, simd_type* a, simd_type* b, simd_type* c, simd_type* d)
    {
        __m128i u0, u1, u2, u3;
        if constexpr (mode == fyx::simd::detail::interleave_load_mode::aligned)
        {
            u0 = _mm_load_si128(reinterpret_cast<const __m128i*>(reinterpret_cast<const simd_type::scalar_t*>(ptr) + (simd_type::lane_width * 0)));
            u1 = _mm_load_si128(reinterpret_cast<const __m128i*>(reinterpret_cast<const simd_type::scalar_t*>(ptr) + (simd_type::lane_width * 1)));
            u2 = _mm_load_si128(reinterpret_cast<const __m128i*>(reinterpret_cast<const simd_type::scalar_t*>(ptr) + (simd_type::lane_width * 2)));
            u3 = _mm_load_si128(reinterpret_cast<const __m128i*>(reinterpret_cast<const simd_type::scalar_t*>(ptr) + (simd_type::lane_width * 3)));
        }
        else
        {
            u0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(reinterpret_cast<const simd_type::scalar_t*>(ptr) + (simd_type::lane_width * 0)));
            u1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(reinterpret_cast<const simd_type::scalar_t*>(ptr) + (simd_type::lane_width * 1)));
            u2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(reinterpret_cast<const simd_type::scalar_t*>(ptr) + (simd_type::lane_width * 2)));
            u3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(reinterpret_cast<const simd_type::scalar_t*>(ptr) + (simd_type::lane_width * 3)));
        }

        __m128i v0 = _mm_unpacklo_epi16(u0, u2);
        __m128i v1 = _mm_unpackhi_epi16(u0, u2);
        __m128i v2 = _mm_unpacklo_epi16(u1, u3);
        __m128i v3 = _mm_unpackhi_epi16(u1, u3);

        u0 = _mm_unpacklo_epi16(v0, v2);
        u1 = _mm_unpacklo_epi16(v1, v3);
        u2 = _mm_unpackhi_epi16(v0, v2);
        u3 = _mm_unpackhi_epi16(v1, v3);

        (*a).data = _mm_unpacklo_epi16(u0, u1);
        (*b).data = _mm_unpackhi_epi16(u0, u1);
        (*c).data = _mm_unpacklo_epi16(u2, u3);
        (*d).data = _mm_unpackhi_epi16(u2, u3);
    }

    template<typename simd_type, fyx::simd::detail::interleave_load_mode mode>
        requires((sizeof(typename simd_type::scalar_t) == sizeof(std::uint32_t))
    && fyx::simd::is_128bits_simd_v<simd_type>)
        void load_interleave_impl_lane2(const void* ptr, simd_type* a, simd_type* b)
    {
        __m128i v0, v1;
        if constexpr (mode == fyx::simd::detail::interleave_load_mode::aligned)
        {
            v0 = _mm_load_si128(reinterpret_cast<const __m128i*>(reinterpret_cast<const simd_type::scalar_t*>(ptr) + (simd_type::lane_width * 0)));
            v1 = _mm_load_si128(reinterpret_cast<const __m128i*>(reinterpret_cast<const simd_type::scalar_t*>(ptr) + (simd_type::lane_width * 1)));
        }
        else
        {
            v0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(reinterpret_cast<const simd_type::scalar_t*>(ptr) + (simd_type::lane_width * 0)));
            v1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(reinterpret_cast<const simd_type::scalar_t*>(ptr) + (simd_type::lane_width * 1)));
        }

        __m128i v2 = _mm_unpacklo_epi32(v0, v1);
        __m128i v3 = _mm_unpackhi_epi32(v0, v1);

        (*a).data = _mm_unpacklo_epi32(v2, v3);
        (*b).data = _mm_unpackhi_epi32(v2, v3);
    }

    template<typename simd_type, fyx::simd::detail::interleave_load_mode mode>
        requires((sizeof(typename simd_type::scalar_t) == sizeof(std::uint32_t))
    && fyx::simd::is_128bits_simd_v<simd_type>)
        void load_interleave_impl_lane3(const void* ptr, simd_type* a, simd_type* b, simd_type* c)
    {
        __m128i t00, t01, t02;
        if constexpr (mode == fyx::simd::detail::interleave_load_mode::aligned)
        {
            t00 = _mm_load_si128(reinterpret_cast<const __m128i*>(reinterpret_cast<const simd_type::scalar_t*>(ptr) + (simd_type::lane_width * 0)));
            t01 = _mm_load_si128(reinterpret_cast<const __m128i*>(reinterpret_cast<const simd_type::scalar_t*>(ptr) + (simd_type::lane_width * 1)));
            t02 = _mm_load_si128(reinterpret_cast<const __m128i*>(reinterpret_cast<const simd_type::scalar_t*>(ptr) + (simd_type::lane_width * 2)));
        }
        else
        {
            t00 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(reinterpret_cast<const simd_type::scalar_t*>(ptr) + (simd_type::lane_width * 0)));
            t01 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(reinterpret_cast<const simd_type::scalar_t*>(ptr) + (simd_type::lane_width * 1)));
            t02 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(reinterpret_cast<const simd_type::scalar_t*>(ptr) + (simd_type::lane_width * 2)));
        }

        __m128i t10 = _mm_unpacklo_epi32(t00, _mm_unpackhi_epi64(t01, t01));
        __m128i t11 = _mm_unpacklo_epi32(_mm_unpackhi_epi64(t00, t00), t02);
        __m128i t12 = _mm_unpacklo_epi32(t01, _mm_unpackhi_epi64(t02, t02));

        (*a).data = _mm_unpacklo_epi32(t10, _mm_unpackhi_epi64(t11, t11));
        (*b).data = _mm_unpacklo_epi32(_mm_unpackhi_epi64(t10, t10), t12);
        (*c).data = _mm_unpacklo_epi32(t11, _mm_unpackhi_epi64(t12, t12));
    }

    template<typename simd_type, fyx::simd::detail::interleave_load_mode mode>
        requires((sizeof(typename simd_type::scalar_t) == sizeof(std::uint32_t))
    && fyx::simd::is_128bits_simd_v<simd_type>)
        void load_interleave_impl_lane4(const void* ptr, simd_type* a, simd_type* b, simd_type* c, simd_type* d)
    {
        __m128i v0, v1, v2, v3;
        if constexpr (mode == fyx::simd::detail::interleave_load_mode::aligned)
        {
            v0 = _mm_load_si128(reinterpret_cast<const __m128i*>(reinterpret_cast<const simd_type::scalar_t*>(ptr) + (simd_type::lane_width * 0)));
            v1 = _mm_load_si128(reinterpret_cast<const __m128i*>(reinterpret_cast<const simd_type::scalar_t*>(ptr) + (simd_type::lane_width * 1)));
            v2 = _mm_load_si128(reinterpret_cast<const __m128i*>(reinterpret_cast<const simd_type::scalar_t*>(ptr) + (simd_type::lane_width * 2)));
            v3 = _mm_load_si128(reinterpret_cast<const __m128i*>(reinterpret_cast<const simd_type::scalar_t*>(ptr) + (simd_type::lane_width * 3)));
        }
        else
        {
            v0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(reinterpret_cast<const simd_type::scalar_t*>(ptr) + (simd_type::lane_width * 0)));
            v1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(reinterpret_cast<const simd_type::scalar_t*>(ptr) + (simd_type::lane_width * 1)));
            v2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(reinterpret_cast<const simd_type::scalar_t*>(ptr) + (simd_type::lane_width * 2)));
            v3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(reinterpret_cast<const simd_type::scalar_t*>(ptr) + (simd_type::lane_width * 3)));
        }

        __m128i t0 = (_mm_unpacklo_epi32(v0, v1));
        __m128i t1 = (_mm_unpacklo_epi32(v2, v3));
        __m128i t2 = (_mm_unpackhi_epi32(v0, v1));
        __m128i t3 = (_mm_unpackhi_epi32(v2, v3));

        (*a).data = (_mm_unpacklo_epi64(t0, t1));
        (*b).data = (_mm_unpackhi_epi64(t0, t1));
        (*c).data = (_mm_unpacklo_epi64(t2, t3));
        (*d).data = (_mm_unpackhi_epi64(t2, t3));
    }


    template<typename simd_type, fyx::simd::detail::interleave_load_mode mode>
        requires((sizeof(typename simd_type::scalar_t) == sizeof(std::uint64_t))
    && fyx::simd::is_128bits_simd_v<simd_type>)
        void load_interleave_impl_lane2(const void* ptr, simd_type* a, simd_type* b)
    {
        __m128i t0, t1;
        if constexpr (mode == fyx::simd::detail::interleave_load_mode::aligned)
        {
            t0 = _mm_load_si128(reinterpret_cast<const __m128i*>(reinterpret_cast<const simd_type::scalar_t*>(ptr) + (simd_type::lane_width * 0)));
            t1 = _mm_load_si128(reinterpret_cast<const __m128i*>(reinterpret_cast<const simd_type::scalar_t*>(ptr) + (simd_type::lane_width * 1)));
        }
        else
        {
            t0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(reinterpret_cast<const simd_type::scalar_t*>(ptr) + (simd_type::lane_width * 0)));
            t1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(reinterpret_cast<const simd_type::scalar_t*>(ptr) + (simd_type::lane_width * 1)));
        }

        (*a).data = _mm_unpacklo_epi64(t0, t1);
        (*b).data = _mm_unpackhi_epi64(t0, t1);
    }

    template<typename simd_type, fyx::simd::detail::interleave_load_mode mode>
        requires((sizeof(typename simd_type::scalar_t) == sizeof(std::uint64_t))
    && fyx::simd::is_128bits_simd_v<simd_type>)
        void load_interleave_impl_lane3(const void* ptr, simd_type* a, simd_type* b, simd_type* c)
    {
        __m128i v0, v1, v2;
        if constexpr (mode == fyx::simd::detail::interleave_load_mode::aligned)
        {
            v0 = _mm_load_si128(reinterpret_cast<const __m128i*>(reinterpret_cast<const simd_type::scalar_t*>(ptr) + (simd_type::lane_width * 0)));
            v1 = _mm_load_si128(reinterpret_cast<const __m128i*>(reinterpret_cast<const simd_type::scalar_t*>(ptr) + (simd_type::lane_width * 1)));
            v2 = _mm_load_si128(reinterpret_cast<const __m128i*>(reinterpret_cast<const simd_type::scalar_t*>(ptr) + (simd_type::lane_width * 2)));
        }
        else
        {
            v0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(reinterpret_cast<const simd_type::scalar_t*>(ptr) + (simd_type::lane_width * 0)));
            v1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(reinterpret_cast<const simd_type::scalar_t*>(ptr) + (simd_type::lane_width * 1)));
            v2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(reinterpret_cast<const simd_type::scalar_t*>(ptr) + (simd_type::lane_width * 2)));
        }

        v1 = _mm_shuffle_epi32(v1, 0x4e);
        (*a).data = _mm_unpacklo_epi64(v0, v1);
        (*b).data = _mm_unpacklo_epi64(_mm_unpackhi_epi64(v0, v0), v2);
        (*c).data = _mm_unpackhi_epi64(v1, v2);
    }

    template<typename simd_type, fyx::simd::detail::interleave_load_mode mode>
        requires((sizeof(typename simd_type::scalar_t) == sizeof(std::uint64_t))
    && fyx::simd::is_128bits_simd_v<simd_type>)
        void load_interleave_impl_lane4(const void* ptr, simd_type* a, simd_type* b, simd_type* c, simd_type* d)
    {
        __m128i t0, t1, t2, t3;
        if constexpr (mode == fyx::simd::detail::interleave_load_mode::aligned)
        {
            t0 = _mm_load_si128(reinterpret_cast<const __m128i*>(reinterpret_cast<const simd_type::scalar_t*>(ptr) + (simd_type::lane_width * 0)));
            t1 = _mm_load_si128(reinterpret_cast<const __m128i*>(reinterpret_cast<const simd_type::scalar_t*>(ptr) + (simd_type::lane_width * 1)));
            t2 = _mm_load_si128(reinterpret_cast<const __m128i*>(reinterpret_cast<const simd_type::scalar_t*>(ptr) + (simd_type::lane_width * 2)));
            t3 = _mm_load_si128(reinterpret_cast<const __m128i*>(reinterpret_cast<const simd_type::scalar_t*>(ptr) + (simd_type::lane_width * 3)));
        }
        else
        {
            t0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(reinterpret_cast<const simd_type::scalar_t*>(ptr) + (simd_type::lane_width * 0)));
            t1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(reinterpret_cast<const simd_type::scalar_t*>(ptr) + (simd_type::lane_width * 1)));
            t2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(reinterpret_cast<const simd_type::scalar_t*>(ptr) + (simd_type::lane_width * 2)));
            t3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(reinterpret_cast<const simd_type::scalar_t*>(ptr) + (simd_type::lane_width * 3)));
        }

        (*a).data = _mm_unpacklo_epi64(t0, t2);
        (*b).data = _mm_unpackhi_epi64(t0, t2);
        (*c).data = _mm_unpacklo_epi64(t1, t3);
        (*d).data = _mm_unpackhi_epi64(t1, t3);
    }

    template<typename simd_type, fyx::simd::detail::interleave_load_mode mode> requires((sizeof(typename simd_type::scalar_t) == sizeof(std::uint8_t))
        && fyx::simd::is_256bits_simd_v<simd_type>)
        void load_interleave_impl_lane2(const typename simd_type::scalar_t* src_scalar_ptr, simd_type* dst_vec0, simd_type* dst_vec1)
    {
        __m256i ab0, ab1;
        if constexpr (mode == fyx::simd::detail::interleave_load_mode::aligned)
        {
            ab0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (simd_type::lane_width * 0)));
            ab1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (simd_type::lane_width * 1)));
        }
        else
        {
            ab0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (simd_type::lane_width * 0)));
            ab1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (simd_type::lane_width * 1)));
        }

        const __m256i shuffle_mask = _mm256_setr_epi8(
            0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15,
            0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15
        );

        __m256i p0 = _mm256_shuffle_epi8(ab0, shuffle_mask);
        __m256i p1 = _mm256_shuffle_epi8(ab1, shuffle_mask);

        __m256i pl = _mm256_permute2x128_si256(p0, p1, 0 + 2 * 16);
        __m256i ph = _mm256_permute2x128_si256(p0, p1, 1 + 3 * 16);

        (*dst_vec0).data = _mm256_unpacklo_epi64(pl, ph);
        (*dst_vec1).data = _mm256_unpackhi_epi64(pl, ph);
    }

    template<typename simd_type, fyx::simd::detail::interleave_load_mode mode> requires((sizeof(typename simd_type::scalar_t) == sizeof(std::uint8_t))
        && fyx::simd::is_256bits_simd_v<simd_type>)
        void load_interleave_impl_lane3(const typename simd_type::scalar_t* src_scalar_ptr, simd_type* dst_vec0, simd_type* dst_vec1, simd_type* dst_vec2)
    {
        __m256i bgr0, bgr1, bgr2;
        if constexpr (mode == fyx::simd::detail::interleave_load_mode::aligned)
        {
            bgr0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (simd_type::lane_width * 0)));
            bgr1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (simd_type::lane_width * 1)));
            bgr2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (simd_type::lane_width * 2)));
        }
        else
        {
            bgr0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (simd_type::lane_width * 0)));
            bgr1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (simd_type::lane_width * 1)));
            bgr2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (simd_type::lane_width * 2)));
        }

        __m256i s02_low = _mm256_permute2x128_si256(bgr0, bgr2, 0 + 2 * 16);
        __m256i s02_high = _mm256_permute2x128_si256(bgr0, bgr2, 1 + 3 * 16);

        const __m256i m0 = _mm256_setr_epi8(
            0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0,
            0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0);

        const __m256i m1 = _mm256_setr_epi8(
            0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0,
            -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1);

        __m256i b0 = _mm256_blendv_epi8(_mm256_blendv_epi8(s02_low, s02_high, m0), bgr1, m1);
        __m256i g0 = _mm256_blendv_epi8(_mm256_blendv_epi8(s02_high, s02_low, m1), bgr1, m0);
        __m256i r0 = _mm256_blendv_epi8(_mm256_blendv_epi8(bgr1, s02_low, m0), s02_high, m1);

        const __m256i sh_b = _mm256_setr_epi8(
            0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14, 1, 4, 7, 10, 13,
            0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14, 1, 4, 7, 10, 13);

        const __m256i sh_g = _mm256_setr_epi8(
            1, 4, 7, 10, 13, 0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14,
            1, 4, 7, 10, 13, 0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14);

        const __m256i sh_r = _mm256_setr_epi8(
            2, 5, 8, 11, 14, 1, 4, 7, 10, 13, 0, 3, 6, 9, 12, 15,
            2, 5, 8, 11, 14, 1, 4, 7, 10, 13, 0, 3, 6, 9, 12, 15);

        (*dst_vec0).data = _mm256_shuffle_epi8(b0, sh_b);
        (*dst_vec1).data = _mm256_shuffle_epi8(g0, sh_g);
        (*dst_vec2).data = _mm256_shuffle_epi8(r0, sh_r);
    }


    template<typename simd_type, fyx::simd::detail::interleave_load_mode mode> requires((sizeof(typename simd_type::scalar_t) == sizeof(std::uint8_t))
        && fyx::simd::is_256bits_simd_v<simd_type>)
        void load_interleave_impl_lane4(const typename simd_type::scalar_t* src_scalar_ptr, simd_type* dst_vec0, simd_type* dst_vec1, simd_type* dst_vec2, simd_type* dst_vec3)
    {
        __m256i bgr0, bgr1, bgr2, bgr3;
        if constexpr (mode == fyx::simd::detail::interleave_load_mode::aligned)
        {
            bgr0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (simd_type::lane_width * 0)));
            bgr1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (simd_type::lane_width * 1)));
            bgr2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (simd_type::lane_width * 2)));
            bgr3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (simd_type::lane_width * 3)));
        }
        else
        {
            bgr0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (simd_type::lane_width * 0)));
            bgr1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (simd_type::lane_width * 1)));
            bgr2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (simd_type::lane_width * 2)));
            bgr3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (simd_type::lane_width * 3)));
        }

        const __m256i mask = _mm256_setr_epi8(
            0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15,
            0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15
        );

        __m256i p0 = _mm256_shuffle_epi8(bgr0, mask);
        __m256i p1 = _mm256_shuffle_epi8(bgr1, mask);
        __m256i p2 = _mm256_shuffle_epi8(bgr2, mask);
        __m256i p3 = _mm256_shuffle_epi8(bgr3, mask);

        __m256i p01l = _mm256_unpacklo_epi32(p0, p1);
        __m256i p01h = _mm256_unpackhi_epi32(p0, p1);
        __m256i p23l = _mm256_unpacklo_epi32(p2, p3);
        __m256i p23h = _mm256_unpackhi_epi32(p2, p3);

        __m256i pll = _mm256_permute2x128_si256(p01l, p23l, 0 + 2 * 16);
        __m256i plh = _mm256_permute2x128_si256(p01l, p23l, 1 + 3 * 16);
        __m256i phl = _mm256_permute2x128_si256(p01h, p23h, 0 + 2 * 16);
        __m256i phh = _mm256_permute2x128_si256(p01h, p23h, 1 + 3 * 16);

        (*dst_vec0).data = _mm256_unpacklo_epi32(pll, plh);
        (*dst_vec1).data = _mm256_unpackhi_epi32(pll, plh);
        (*dst_vec2).data = _mm256_unpacklo_epi32(phl, phh);
        (*dst_vec3).data = _mm256_unpackhi_epi32(phl, phh);
    }


    template<typename simd_type, fyx::simd::detail::interleave_load_mode mode> requires((sizeof(typename simd_type::scalar_t) == sizeof(std::uint16_t))
        && fyx::simd::is_256bits_simd_v<simd_type>)
        void load_interleave_impl_lane2(const typename simd_type::scalar_t* src_scalar_ptr, simd_type* dst_vec0, simd_type* dst_vec1)
    {
        __m256i ab0, ab1;
        if constexpr (mode == fyx::simd::detail::interleave_load_mode::aligned)
        {
            ab0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (simd_type::lane_width * 0)));
            ab1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (simd_type::lane_width * 1)));
        }
        else
        {
            ab0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (simd_type::lane_width * 0)));
            ab1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (simd_type::lane_width * 1)));
        }

        const __m256i sh = _mm256_setr_epi8(
            0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15,
            0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15
        );

        __m256i p0 = _mm256_shuffle_epi8(ab0, sh);
        __m256i p1 = _mm256_shuffle_epi8(ab1, sh);
        __m256i pl = _mm256_permute2x128_si256(p0, p1, 0 + 2 * 16);
        __m256i ph = _mm256_permute2x128_si256(p0, p1, 1 + 3 * 16);

        (*dst_vec0).data = _mm256_unpacklo_epi64(pl, ph);
        (*dst_vec1).data = _mm256_unpackhi_epi64(pl, ph);
    }

    template<typename simd_type, fyx::simd::detail::interleave_load_mode mode> requires((sizeof(typename simd_type::scalar_t) == sizeof(std::uint16_t))
        && fyx::simd::is_256bits_simd_v<simd_type>)
        void load_interleave_impl_lane3(const typename simd_type::scalar_t* src_scalar_ptr, simd_type* dst_vec0, simd_type* dst_vec1, simd_type* dst_vec2)
    {
        __m256i bgr0, bgr1, bgr2;
        if constexpr (mode == fyx::simd::detail::interleave_load_mode::aligned)
        {
            bgr0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (simd_type::lane_width * 0)));
            bgr1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (simd_type::lane_width * 1)));
            bgr2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (simd_type::lane_width * 2)));
        }
        else
        {
            bgr0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (simd_type::lane_width * 0)));
            bgr1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (simd_type::lane_width * 1)));
            bgr2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (simd_type::lane_width * 2)));
        }

        __m256i s02_low = _mm256_permute2x128_si256(bgr0, bgr2, 0 + 2 * 16);
        __m256i s02_high = _mm256_permute2x128_si256(bgr0, bgr2, 1 + 3 * 16);

        const __m256i m0 = _mm256_setr_epi8(
            0, 0, -1, -1, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, -1, -1,
            0, 0, 0, 0, -1, -1, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0);

        const __m256i m1 = _mm256_setr_epi8(
            0, 0, 0, 0, -1, -1, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0,
            -1, -1, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, -1, -1, 0, 0);

        __m256i b0 = _mm256_blendv_epi8(_mm256_blendv_epi8(s02_low, s02_high, m0), bgr1, m1);
        __m256i g0 = _mm256_blendv_epi8(_mm256_blendv_epi8(bgr1, s02_low, m0), s02_high, m1);
        __m256i r0 = _mm256_blendv_epi8(_mm256_blendv_epi8(s02_high, s02_low, m1), bgr1, m0);

        const __m256i sh_b = _mm256_setr_epi8(
            0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15, 4, 5, 10, 11,
            0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15, 4, 5, 10, 11);

        const __m256i sh_g = _mm256_setr_epi8(
            2, 3, 8, 9, 14, 15, 4, 5, 10, 11, 0, 1, 6, 7, 12, 13,
            2, 3, 8, 9, 14, 15, 4, 5, 10, 11, 0, 1, 6, 7, 12, 13);

        const __m256i sh_r = _mm256_setr_epi8(
            4, 5, 10, 11, 0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15,
            4, 5, 10, 11, 0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15);

        (*dst_vec0).data = _mm256_shuffle_epi8(b0, sh_b);
        (*dst_vec1).data = _mm256_shuffle_epi8(g0, sh_g);
        (*dst_vec2).data = _mm256_shuffle_epi8(r0, sh_r);
    }

    template<typename simd_type, fyx::simd::detail::interleave_load_mode mode> requires((sizeof(typename simd_type::scalar_t) == sizeof(std::uint16_t))
        && fyx::simd::is_256bits_simd_v<simd_type>)
        void load_interleave_impl_lane4(const typename simd_type::scalar_t* src_scalar_ptr, simd_type* dst_vec0, simd_type* dst_vec1, simd_type* dst_vec2, simd_type* dst_vec3)
    {
        __m256i bgr0, bgr1, bgr2, bgr3;
        if constexpr (mode == fyx::simd::detail::interleave_load_mode::aligned)
        {
            bgr0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (simd_type::lane_width * 0)));
            bgr1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (simd_type::lane_width * 1)));
            bgr2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (simd_type::lane_width * 2)));
            bgr3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (simd_type::lane_width * 3)));
        }
        else
        {
            bgr0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (simd_type::lane_width * 0)));
            bgr1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (simd_type::lane_width * 1)));
            bgr2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (simd_type::lane_width * 2)));
            bgr3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (simd_type::lane_width * 3)));
        }

        const __m256i sh = _mm256_setr_epi8(
            0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15,
            0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15);

        __m256i p0 = _mm256_shuffle_epi8(bgr0, sh);
        __m256i p1 = _mm256_shuffle_epi8(bgr1, sh);
        __m256i p2 = _mm256_shuffle_epi8(bgr2, sh);
        __m256i p3 = _mm256_shuffle_epi8(bgr3, sh);

        __m256i p01l = _mm256_unpacklo_epi32(p0, p1);
        __m256i p01h = _mm256_unpackhi_epi32(p0, p1);
        __m256i p23l = _mm256_unpacklo_epi32(p2, p3);
        __m256i p23h = _mm256_unpackhi_epi32(p2, p3);

        __m256i pll = _mm256_permute2x128_si256(p01l, p23l, 0 + 2 * 16);
        __m256i plh = _mm256_permute2x128_si256(p01l, p23l, 1 + 3 * 16);
        __m256i phl = _mm256_permute2x128_si256(p01h, p23h, 0 + 2 * 16);
        __m256i phh = _mm256_permute2x128_si256(p01h, p23h, 1 + 3 * 16);

        (*dst_vec0).data = _mm256_unpacklo_epi32(pll, plh);
        (*dst_vec1).data = _mm256_unpackhi_epi32(pll, plh);
        (*dst_vec2).data = _mm256_unpacklo_epi32(phl, phh);
        (*dst_vec3).data = _mm256_unpackhi_epi32(phl, phh);
    }

    template<typename simd_type, fyx::simd::detail::interleave_load_mode mode> requires((sizeof(typename simd_type::scalar_t) == sizeof(std::uint32_t))
        && fyx::simd::is_256bits_simd_v<simd_type>)
        void load_interleave_impl_lane2(const typename simd_type::scalar_t* src_scalar_ptr, simd_type* dst_vec0, simd_type* dst_vec1)
    {
        __m256i ab0, ab1;
        if constexpr (mode == fyx::simd::detail::interleave_load_mode::aligned)
        {
            __m256i ab0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (simd_type::lane_width * 0)));
            __m256i ab1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (simd_type::lane_width * 1)));
        }
        else
        {
            __m256i ab0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (simd_type::lane_width * 0)));
            __m256i ab1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (simd_type::lane_width * 1)));
        }

        constexpr int sh = 0 + 2 * 4 + 1 * 16 + 3 * 64;
        __m256i p0 = _mm256_shuffle_epi32(ab0, sh);
        __m256i p1 = _mm256_shuffle_epi32(ab1, sh);

        __m256i pl = _mm256_permute2x128_si256(p0, p1, 0 + 2 * 16);
        __m256i ph = _mm256_permute2x128_si256(p0, p1, 1 + 3 * 16);

        (*dst_vec0).data = _mm256_unpacklo_epi64(pl, ph);
        (*dst_vec1).data = _mm256_unpackhi_epi64(pl, ph);
    }

    template<typename simd_type, fyx::simd::detail::interleave_load_mode mode> requires((sizeof(typename simd_type::scalar_t) == sizeof(std::uint32_t))
        && fyx::simd::is_256bits_simd_v<simd_type>)
        void load_interleave_impl_lane3(const typename simd_type::scalar_t* src_scalar_ptr, simd_type* dst_vec0, simd_type* dst_vec1, simd_type* dst_vec2)
    {
        __m256i bgr0, bgr1, bgr2;
        if constexpr (mode == fyx::simd::detail::interleave_load_mode::aligned)
        {
            bgr0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (simd_type::lane_width * 0)));
            bgr1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (simd_type::lane_width * 1)));
            bgr2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (simd_type::lane_width * 2)));
        }
        else
        {
            bgr0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (simd_type::lane_width * 0)));
            bgr1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (simd_type::lane_width * 1)));
            bgr2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (simd_type::lane_width * 2)));
        }

        __m256i s02_low = _mm256_permute2x128_si256(bgr0, bgr2, 0 + 2 * 16);
        __m256i s02_high = _mm256_permute2x128_si256(bgr0, bgr2, 1 + 3 * 16);

        __m256i b0 = _mm256_blend_epi32(_mm256_blend_epi32(s02_low, s02_high, 0x24), bgr1, 0x92);
        __m256i g0 = _mm256_blend_epi32(_mm256_blend_epi32(s02_high, s02_low, 0x92), bgr1, 0x24);
        __m256i r0 = _mm256_blend_epi32(_mm256_blend_epi32(bgr1, s02_low, 0x24), s02_high, 0x92);

        (*dst_vec0).data = _mm256_shuffle_epi32(b0, 0x6c);
        (*dst_vec1).data = _mm256_shuffle_epi32(g0, 0xb1);
        (*dst_vec2).data = _mm256_shuffle_epi32(r0, 0xc6);
    }

    template<typename simd_type, fyx::simd::detail::interleave_load_mode mode> requires((sizeof(typename simd_type::scalar_t) == sizeof(std::uint32_t))
        && fyx::simd::is_256bits_simd_v<simd_type>)
        void load_interleave_impl_lane4(const typename simd_type::scalar_t* src_scalar_ptr, simd_type* dst_vec0, simd_type* dst_vec1, simd_type* dst_vec2, simd_type* dst_vec3)
    {
        __m256i p0, p1, p2, p3;
        if constexpr (mode == fyx::simd::detail::interleave_load_mode::aligned)
        {
            p0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (simd_type::lane_width * 0)));
            p1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (simd_type::lane_width * 1)));
            p2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (simd_type::lane_width * 2)));
            p3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (simd_type::lane_width * 3)));
        }
        else
        {
            p0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (simd_type::lane_width * 0)));
            p1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (simd_type::lane_width * 1)));
            p2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (simd_type::lane_width * 2)));
            p3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (simd_type::lane_width * 3)));
        }

        __m256i p01l = _mm256_unpacklo_epi32(p0, p1);
        __m256i p01h = _mm256_unpackhi_epi32(p0, p1);
        __m256i p23l = _mm256_unpacklo_epi32(p2, p3);
        __m256i p23h = _mm256_unpackhi_epi32(p2, p3);

        __m256i pll = _mm256_permute2x128_si256(p01l, p23l, 0 + 2 * 16);
        __m256i plh = _mm256_permute2x128_si256(p01l, p23l, 1 + 3 * 16);
        __m256i phl = _mm256_permute2x128_si256(p01h, p23h, 0 + 2 * 16);
        __m256i phh = _mm256_permute2x128_si256(p01h, p23h, 1 + 3 * 16);

        (*dst_vec0).data = _mm256_unpacklo_epi32(pll, plh);
        (*dst_vec1).data = _mm256_unpackhi_epi32(pll, plh);
        (*dst_vec2).data = _mm256_unpacklo_epi32(phl, phh);
        (*dst_vec3).data = _mm256_unpackhi_epi32(phl, phh);
    }


    template<typename simd_type, fyx::simd::detail::interleave_load_mode mode> requires((sizeof(typename simd_type::scalar_t) == sizeof(std::uint64_t))
        && fyx::simd::is_256bits_simd_v<simd_type>)
        void load_interleave_impl_lane2(const typename simd_type::scalar_t* src_scalar_ptr, simd_type* dst_vec0, simd_type* dst_vec1)
    {
        __m256i ab0, ab1;
        if constexpr (mode == fyx::simd::detail::interleave_load_mode::aligned)
        {
            ab0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (simd_type::lane_width * 0)));
            ab1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (simd_type::lane_width * 1)));
        }
        else
        {
            ab0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (simd_type::lane_width * 0)));
            ab1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (simd_type::lane_width * 1)));
        }

        __m256i pl = _mm256_permute2x128_si256(ab0, ab1, 0 + 2 * 16);
        __m256i ph = _mm256_permute2x128_si256(ab0, ab1, 1 + 3 * 16);

        (*dst_vec0).data = _mm256_unpacklo_epi64(pl, ph);
        (*dst_vec1).data = _mm256_unpackhi_epi64(pl, ph);
    }

    template<typename simd_type, fyx::simd::detail::interleave_load_mode mode> requires((sizeof(typename simd_type::scalar_t) == sizeof(std::uint64_t))
        && fyx::simd::is_256bits_simd_v<simd_type>)
        void load_interleave_impl_lane3(const typename simd_type::scalar_t* src_scalar_ptr, simd_type* dst_vec0, simd_type* dst_vec1, simd_type* dst_vec2)
    {
        __m256i bgr0, bgr1, bgr2;
        if constexpr (mode == fyx::simd::detail::interleave_load_mode::aligned)
        {
            bgr0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (simd_type::lane_width * 0)));
            bgr1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (simd_type::lane_width * 1)));
            bgr2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (simd_type::lane_width * 2)));
        }
        else
        {
            bgr0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (simd_type::lane_width * 0)));
            bgr1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (simd_type::lane_width * 1)));
            bgr2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (simd_type::lane_width * 2)));
        }

        __m256i s01 = _mm256_blend_epi32(bgr0, bgr1, 0xf0);
        __m256i s12 = _mm256_blend_epi32(bgr1, bgr2, 0xf0);
        __m256i s20r = _mm256_permute4x64_epi64(_mm256_blend_epi32(bgr2, bgr0, 0xf0), 0x1b);

        (*dst_vec0).data = _mm256_unpacklo_epi64(s01, s20r);
        (*dst_vec1).data = _mm256_alignr_epi8(s12, s01, 8);
        (*dst_vec2).data = _mm256_unpackhi_epi64(s20r, s12);
    }

    template<typename simd_type, fyx::simd::detail::interleave_load_mode mode> requires((sizeof(typename simd_type::scalar_t) == sizeof(std::uint64_t))
        && fyx::simd::is_256bits_simd_v<simd_type>)
        void load_interleave_impl_lane4(const typename simd_type::scalar_t* src_scalar_ptr, simd_type* dst_vec0, simd_type* dst_vec1, simd_type* dst_vec2, simd_type* dst_vec3)
    {
        __m256i bgra0, bgra1, bgra2, bgra3;
        if constexpr (mode == fyx::simd::detail::interleave_load_mode::aligned)
        {
            __m256i bgra0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (simd_type::lane_width * 0)));
            __m256i bgra1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (simd_type::lane_width * 1)));
            __m256i bgra2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (simd_type::lane_width * 2)));
            __m256i bgra3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (simd_type::lane_width * 3)));
        }
        else
        {
            __m256i bgra0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (simd_type::lane_width * 0)));
            __m256i bgra1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (simd_type::lane_width * 1)));
            __m256i bgra2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (simd_type::lane_width * 2)));
            __m256i bgra3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(src_scalar_ptr + (simd_type::lane_width * 3)));
        }

        __m256i l02 = _mm256_permute2x128_si256(bgra0, bgra2, 0 + 2 * 16);
        __m256i h02 = _mm256_permute2x128_si256(bgra0, bgra2, 1 + 3 * 16);
        __m256i l13 = _mm256_permute2x128_si256(bgra1, bgra3, 0 + 2 * 16);
        __m256i h13 = _mm256_permute2x128_si256(bgra1, bgra3, 1 + 3 * 16);

        (*dst_vec0).data = _mm256_unpacklo_epi64(l02, l13);
        (*dst_vec1).data = _mm256_unpackhi_epi64(l02, l13);
        (*dst_vec2).data = _mm256_unpacklo_epi64(h02, h13);
        (*dst_vec3).data = _mm256_unpackhi_epi64(h02, h13);
    }



    template<typename simd_type, fyx::simd::detail::interleave_store_mode mode>
        requires((sizeof(typename simd_type::scalar_t) == sizeof(std::uint8_t))
    && fyx::simd::is_128bits_simd_v<simd_type>)
        void store_interleave_impl_lane2(void* mem_addr, simd_type a, simd_type b)
    {
        __m128i v0 = _mm_unpacklo_epi8(a.data, b.data);
        __m128i v1 = _mm_unpackhi_epi8(a.data, b.data);

        if (mode == fyx::simd::detail::interleave_store_mode::stream)
        {
            _mm_stream_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 0)), v0);
            _mm_stream_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 1)), v1);
        }
        else if (mode == fyx::simd::detail::interleave_store_mode::aligned)
        {
            _mm_store_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 0)), v0);
            _mm_store_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 1)), v1);
        }
        else
        {
            _mm_storeu_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 0)), v0);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 1)), v1);
        }
    }

    template<typename simd_type, fyx::simd::detail::interleave_store_mode mode>
        requires((sizeof(typename simd_type::scalar_t) == sizeof(std::uint8_t))
    && fyx::simd::is_128bits_simd_v<simd_type>)
        void store_interleave_impl_lane3(void* mem_addr, simd_type a, simd_type b, simd_type c)
    {
        const __m128i sh_a = _mm_setr_epi8(0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10, 5);
        const __m128i sh_b = _mm_setr_epi8(5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10);
        const __m128i sh_c = _mm_setr_epi8(10, 5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15);

        __m128i a0 = _mm_shuffle_epi8(a.data, sh_a);
        __m128i b0 = _mm_shuffle_epi8(b.data, sh_b);
        __m128i c0 = _mm_shuffle_epi8(c.data, sh_c);

        const __m128i m0 = _mm_setr_epi8(0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0);
        const __m128i m1 = _mm_setr_epi8(0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0);

        __m128i v0 = _mm_blendv_epi8(_mm_blendv_epi8(a0, b0, m1), c0, m0);
        __m128i v1 = _mm_blendv_epi8(_mm_blendv_epi8(b0, c0, m1), a0, m0);
        __m128i v2 = _mm_blendv_epi8(_mm_blendv_epi8(c0, a0, m1), b0, m0);

        if (mode == fyx::simd::detail::interleave_store_mode::stream)
        {
            _mm_stream_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 0)), v0);
            _mm_stream_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 1)), v1);
            _mm_stream_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 2)), v2);
        }
        else if (mode == fyx::simd::detail::interleave_store_mode::aligned)
        {
            _mm_store_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 0)), v0);
            _mm_store_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 1)), v1);
            _mm_store_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 2)), v2);
        }
        else
        {
            _mm_storeu_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 0)), v0);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 1)), v1);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 2)), v2);
        }
    }

    template<typename simd_type, fyx::simd::detail::interleave_store_mode mode>
        requires((sizeof(typename simd_type::scalar_t) == sizeof(std::uint8_t))
    && fyx::simd::is_128bits_simd_v<simd_type>)
        void store_interleave_impl_lane4(void* mem_addr, simd_type a, simd_type b, simd_type c, simd_type d)
    {
        __m128i u0 = _mm_unpacklo_epi8(a.data, c.data);
        __m128i u1 = _mm_unpackhi_epi8(a.data, c.data);
        __m128i u2 = _mm_unpacklo_epi8(b.data, d.data);
        __m128i u3 = _mm_unpackhi_epi8(b.data, d.data);

        __m128i v0 = _mm_unpacklo_epi8(u0, u2);
        __m128i v1 = _mm_unpackhi_epi8(u0, u2);
        __m128i v2 = _mm_unpacklo_epi8(u1, u3);
        __m128i v3 = _mm_unpackhi_epi8(u1, u3);

        if (mode == fyx::simd::detail::interleave_store_mode::stream)
        {
            _mm_stream_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 0)), v0);
            _mm_stream_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 1)), v1);
            _mm_stream_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 2)), v2);
            _mm_stream_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 3)), v3);
        }
        else if (mode == fyx::simd::detail::interleave_store_mode::aligned)
        {
            _mm_store_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 0)), v0);
            _mm_store_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 1)), v1);
            _mm_store_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 2)), v2);
            _mm_store_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 3)), v3);
        }
        else
        {
            _mm_storeu_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 0)), v0);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 1)), v1);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 2)), v2);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 3)), v3);
        }
    }

    template<typename simd_type, fyx::simd::detail::interleave_store_mode mode>
        requires((sizeof(typename simd_type::scalar_t) == sizeof(std::uint16_t))
    && fyx::simd::is_128bits_simd_v<simd_type>)
        void store_interleave_impl_lane2(void* mem_addr, simd_type a, simd_type b)
    {
        __m128i v0 = _mm_unpacklo_epi16(a.data, b.data);
        __m128i v1 = _mm_unpackhi_epi16(a.data, b.data);

        if (mode == fyx::simd::detail::interleave_store_mode::stream)
        {
            _mm_stream_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 0)), v0);
            _mm_stream_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 1)), v1);
        }
        else if (mode == fyx::simd::detail::interleave_store_mode::aligned)
        {
            _mm_store_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 0)), v0);
            _mm_store_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 1)), v1);
        }
        else
        {
            _mm_storeu_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 0)), v0);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 1)), v1);
        }
    }


    template<typename simd_type, fyx::simd::detail::interleave_store_mode mode>
        requires((sizeof(typename simd_type::scalar_t) == sizeof(std::uint16_t))
    && fyx::simd::is_128bits_simd_v<simd_type>)
        void store_interleave_impl_lane3(void* mem_addr, simd_type a, simd_type b, simd_type c)
    {
        const __m128i sh_a = _mm_setr_epi8(0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15, 4, 5, 10, 11);
        const __m128i sh_b = _mm_setr_epi8(10, 11, 0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15, 4, 5);
        const __m128i sh_c = _mm_setr_epi8(4, 5, 10, 11, 0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15);

        __m128i a0 = _mm_shuffle_epi8(a.data, sh_a);
        __m128i b0 = _mm_shuffle_epi8(b.data, sh_b);
        __m128i c0 = _mm_shuffle_epi8(c.data, sh_c);

        __m128i v0 = _mm_blend_epi16(_mm_blend_epi16(a0, b0, 0x92), c0, 0x24);
        __m128i v1 = _mm_blend_epi16(_mm_blend_epi16(c0, a0, 0x92), b0, 0x24);
        __m128i v2 = _mm_blend_epi16(_mm_blend_epi16(b0, c0, 0x92), a0, 0x24);

        if (mode == fyx::simd::detail::interleave_store_mode::stream)
        {
            _mm_stream_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 0)), v0);
            _mm_stream_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 1)), v1);
            _mm_stream_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 2)), v2);
        }
        else if (mode == fyx::simd::detail::interleave_store_mode::aligned)
        {
            _mm_store_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 0)), v0);
            _mm_store_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 1)), v1);
            _mm_store_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 2)), v2);
        }
        else
        {
            _mm_storeu_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 0)), v0);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 1)), v1);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 2)), v2);
        }
    }

    template<typename simd_type, fyx::simd::detail::interleave_store_mode mode>
        requires((sizeof(typename simd_type::scalar_t) == sizeof(std::uint16_t))
    && fyx::simd::is_128bits_simd_v<simd_type>)
        void store_interleave_impl_lane4(void* mem_addr, simd_type a, simd_type b, simd_type c, simd_type d)
    {
        __m128i u0 = _mm_unpacklo_epi16(a.data, c.data);
        __m128i u1 = _mm_unpackhi_epi16(a.data, c.data);
        __m128i u2 = _mm_unpacklo_epi16(b.data, d.data);
        __m128i u3 = _mm_unpackhi_epi16(b.data, d.data);

        __m128i v0 = _mm_unpacklo_epi16(u0, u2);
        __m128i v1 = _mm_unpackhi_epi16(u0, u2);
        __m128i v2 = _mm_unpacklo_epi16(u1, u3);
        __m128i v3 = _mm_unpackhi_epi16(u1, u3);

        if (mode == fyx::simd::detail::interleave_store_mode::stream)
        {
            _mm_stream_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 0)), v0);
            _mm_stream_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 1)), v1);
            _mm_stream_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 2)), v2);
            _mm_stream_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 3)), v3);
        }
        else if (mode == fyx::simd::detail::interleave_store_mode::aligned)
        {
            _mm_store_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 0)), v0);
            _mm_store_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 1)), v1);
            _mm_store_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 2)), v2);
            _mm_store_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 3)), v3);
        }
        else
        {
            _mm_storeu_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 0)), v0);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 1)), v1);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 2)), v2);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 3)), v3);
        }
    }


    template<typename simd_type, fyx::simd::detail::interleave_store_mode mode>
        requires((sizeof(typename simd_type::scalar_t) == sizeof(std::uint32_t))
    && fyx::simd::is_128bits_simd_v<simd_type>)
        void store_interleave_impl_lane2(void* mem_addr, simd_type a, simd_type b)
    {
        __m128i v0 = _mm_unpacklo_epi32(a.data, b.data);
        __m128i v1 = _mm_unpackhi_epi32(a.data, b.data);

        if (mode == fyx::simd::detail::interleave_store_mode::stream)
        {
            _mm_stream_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 0)), v0);
            _mm_stream_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 1)), v1);
        }
        else if (mode == fyx::simd::detail::interleave_store_mode::aligned)
        {
            _mm_store_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 0)), v0);
            _mm_store_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 1)), v1);
        }
        else
        {
            _mm_storeu_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 0)), v0);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 1)), v1);
        }
    }

    inline void v_transpose4x4(
        const __m128i& a0, const __m128i& a1, const __m128i& a2, const __m128i& a3,
        __m128i& b0, __m128i& b1, __m128i& b2, __m128i& b3)
    {
        __m128i t0 = (_mm_unpacklo_epi32(a0, a1));
        __m128i t1 = (_mm_unpacklo_epi32(a2, a3));
        __m128i t2 = (_mm_unpackhi_epi32(a0, a1));
        __m128i t3 = (_mm_unpackhi_epi32(a2, a3));
        b0 = (_mm_unpacklo_epi64(t0, t1));
        b1 = (_mm_unpackhi_epi64(t0, t1));
        b2 = (_mm_unpacklo_epi64(t2, t3));
        b3 = (_mm_unpackhi_epi64(t2, t3));
    }

    template<typename simd_type, fyx::simd::detail::interleave_store_mode mode>
        requires((sizeof(typename simd_type::scalar_t) == sizeof(std::uint32_t))
    && fyx::simd::is_128bits_simd_v<simd_type>)
        void store_interleave_impl_lane3(void* mem_addr, simd_type a, simd_type b, simd_type c)
    {
        __m128i z = _mm_setzero_si128();

        __m128i t0 = _mm_unpacklo_epi32(a.data, b.data);
        __m128i t1 = _mm_unpacklo_epi32(c.data, z);
        __m128i t2 = _mm_unpackhi_epi32(a.data, b.data);
        __m128i t3 = _mm_unpackhi_epi32(c.data, z);
        __m128i u0 = _mm_unpacklo_epi64(t0, t1);
        __m128i u1 = _mm_unpackhi_epi64(t0, t1);
        __m128i u2 = _mm_unpacklo_epi64(t2, t3);
        __m128i u3 = _mm_unpackhi_epi64(t2, t3);

        __m128i v0 = _mm_or_si128(u0, _mm_slli_si128(u1, 12));
        __m128i v1 = _mm_or_si128(_mm_srli_si128(u1, 4), _mm_slli_si128(u2, 8));
        __m128i v2 = _mm_or_si128(_mm_srli_si128(u2, 8), _mm_slli_si128(u3, 4));

        if (mode == fyx::simd::detail::interleave_store_mode::stream)
        {
            _mm_stream_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 0)), v0);
            _mm_stream_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 1)), v1);
            _mm_stream_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 2)), v2);
        }
        else if (mode == fyx::simd::detail::interleave_store_mode::aligned)
        {
            _mm_store_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 0)), v0);
            _mm_store_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 1)), v1);
            _mm_store_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 2)), v2);
        }
        else
        {
            _mm_storeu_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 0)), v0);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 1)), v1);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 2)), v2);
        }
    }

    template<typename simd_type, fyx::simd::detail::interleave_store_mode mode>
        requires((sizeof(typename simd_type::scalar_t) == sizeof(std::uint32_t))
    && fyx::simd::is_128bits_simd_v<simd_type>)
        void store_interleave_impl_lane4(void* mem_addr, simd_type a, simd_type b, simd_type c, simd_type d)
    {
        __m128i t0 = _mm_unpacklo_epi32(a.data, b.data);
        __m128i t1 = _mm_unpacklo_epi32(c.data, d.data);
        __m128i t2 = _mm_unpackhi_epi32(a.data, b.data);
        __m128i t3 = _mm_unpackhi_epi32(c.data, d.data);

        __m128i v0 = _mm_unpacklo_epi64(t0, t1);
        __m128i v1 = _mm_unpackhi_epi64(t0, t1);
        __m128i v2 = _mm_unpacklo_epi64(t2, t3);
        __m128i v3 = _mm_unpackhi_epi64(t2, t3);

        if (mode == fyx::simd::detail::interleave_store_mode::stream)
        {
            _mm_stream_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 0)), v0);
            _mm_stream_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 1)), v1);
            _mm_stream_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 2)), v2);
            _mm_stream_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 3)), v3);
        }
        else if (mode == fyx::simd::detail::interleave_store_mode::aligned)
        {
            _mm_store_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 0)), v0);
            _mm_store_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 1)), v1);
            _mm_store_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 2)), v2);
            _mm_store_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 3)), v3);
        }
        else
        {
            _mm_storeu_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 0)), v0);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 1)), v1);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 2)), v2);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 3)), v3);
        }
    }


    template<typename simd_type, fyx::simd::detail::interleave_store_mode mode>
        requires((sizeof(typename simd_type::scalar_t) == sizeof(std::uint64_t))
    && fyx::simd::is_128bits_simd_v<simd_type>)
        void store_interleave_impl_lane2(void* mem_addr, simd_type a, simd_type b)
    {
        __m128i v0 = _mm_unpacklo_epi64(a.data, b.data);
        __m128i v1 = _mm_unpackhi_epi64(a.data, b.data);

        if (mode == fyx::simd::detail::interleave_store_mode::stream)
        {
            _mm_stream_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 0)), v0);
            _mm_stream_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 1)), v1);
        }
        else if (mode == fyx::simd::detail::interleave_store_mode::aligned)
        {
            _mm_store_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 0)), v0);
            _mm_store_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 1)), v1);
        }
        else
        {
            _mm_storeu_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 0)), v0);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 1)), v1);
        }
    }

    template<typename simd_type, fyx::simd::detail::interleave_store_mode mode>
        requires((sizeof(typename simd_type::scalar_t) == sizeof(std::uint64_t))
    && fyx::simd::is_128bits_simd_v<simd_type>)
        void store_interleave_impl_lane3(void* mem_addr, simd_type a, simd_type b, simd_type c)
    {
        __m128i v0 = _mm_unpacklo_epi64(a.data, b.data);
        __m128i v1 = _mm_unpacklo_epi64(c.data, _mm_unpackhi_epi64(a.data, a.data));
        __m128i v2 = _mm_unpackhi_epi64(b.data, c.data);

        if (mode == fyx::simd::detail::interleave_store_mode::stream)
        {
            _mm_stream_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 0)), v0);
            _mm_stream_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 1)), v1);
            _mm_stream_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 2)), v2);
        }
        else if (mode == fyx::simd::detail::interleave_store_mode::aligned)
        {
            _mm_store_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 0)), v0);
            _mm_store_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 1)), v1);
            _mm_store_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 2)), v2);
        }
        else
        {
            _mm_storeu_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 0)), v0);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 1)), v1);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 2)), v2);
        }
    }

    template<typename simd_type, fyx::simd::detail::interleave_store_mode mode>
        requires((sizeof(typename simd_type::scalar_t) == sizeof(std::uint64_t))
    && fyx::simd::is_128bits_simd_v<simd_type>)
        void store_interleave_impl_lane4(void* mem_addr, simd_type a, simd_type b, simd_type c, simd_type d)
    {
        __m128i v0 = _mm_unpacklo_epi64(a.data, b.data);
        __m128i v1 = _mm_unpacklo_epi64(c.data, d.data);
        __m128i v2 = _mm_unpackhi_epi64(a.data, b.data);
        __m128i v3 = _mm_unpackhi_epi64(c.data, d.data);

        if (mode == fyx::simd::detail::interleave_store_mode::stream)
        {
            _mm_stream_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 0)), v0);
            _mm_stream_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 1)), v1);
            _mm_stream_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 2)), v2);
            _mm_stream_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 3)), v3);
        }
        else if (mode == fyx::simd::detail::interleave_store_mode::aligned)
        {
            _mm_store_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 0)), v0);
            _mm_store_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 1)), v1);
            _mm_store_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 2)), v2);
            _mm_store_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 3)), v3);
        }
        else
        {
            _mm_storeu_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 0)), v0);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 1)), v1);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 2)), v2);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 3)), v3);
        }
    }

    template<typename simd_type, fyx::simd::detail::interleave_store_mode mode> requires((sizeof(typename simd_type::scalar_t) == sizeof(std::uint8_t))
        && fyx::simd::is_256bits_simd_v<simd_type>)
        void store_interleave_impl_lane2(void* mem_addr, simd_type v0, simd_type v1)
    {
        __m256i v_lane0 = v0.data;
        __m256i v_lane1 = v1.data;

        __m256i xy_l = _mm256_unpacklo_epi8(v_lane0, v_lane1);
        __m256i xy_h = _mm256_unpackhi_epi8(v_lane0, v_lane1);

        __m256i xy0 = _mm256_permute2x128_si256(xy_l, xy_h, 0 + 2 * 16);
        __m256i xy1 = _mm256_permute2x128_si256(xy_l, xy_h, 1 + 3 * 16);

        typename simd_type::scalar_t* mem_addr_lane_0 = reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 0);
        typename simd_type::scalar_t* mem_addr_lane_1 = reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 1);

        if constexpr (mode == fyx::simd::detail::interleave_store_mode::aligned)
        {
            _mm256_store_si256(reinterpret_cast<__m256i*>((mem_addr_lane_0)), xy0);
            _mm256_store_si256(reinterpret_cast<__m256i*>((mem_addr_lane_1)), xy1);
        }
        else if constexpr (mode == fyx::simd::detail::interleave_store_mode::stream)
        {
            _mm256_stream_si256(reinterpret_cast<__m256i*>((mem_addr_lane_0)), xy0);
            _mm256_stream_si256(reinterpret_cast<__m256i*>((mem_addr_lane_1)), xy1);
        }
        else
        {
            _mm256_storeu_si256(reinterpret_cast<__m256i*>((mem_addr_lane_0)), xy0);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>((mem_addr_lane_1)), xy1);
        }
    }

    template<typename simd_type, fyx::simd::detail::interleave_store_mode mode> requires((sizeof(typename simd_type::scalar_t) == sizeof(std::uint8_t))
        && fyx::simd::is_256bits_simd_v<simd_type>)
        void store_interleave_impl_lane3(void* mem_addr, simd_type v0, simd_type v1, simd_type v2)
    {
        const __m256i sh_b = _mm256_setr_epi8(
            0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10, 5,
            0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10, 5);

        const __m256i sh_g = _mm256_setr_epi8(
            5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10,
            5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10);

        const __m256i sh_r = _mm256_setr_epi8(
            10, 5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15,
            10, 5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15);

        const __m256i m0 = _mm256_setr_epi8(
            0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0,
            0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0);

        const __m256i m1 = _mm256_setr_epi8(
            0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0,
            0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0);

        __m256i v_lane0 = v0.data;
        __m256i v_lane1 = v1.data;
        __m256i v_lane2 = v2.data;

        __m256i b0 = _mm256_shuffle_epi8(v_lane0, sh_b);
        __m256i g0 = _mm256_shuffle_epi8(v_lane1, sh_g);
        __m256i r0 = _mm256_shuffle_epi8(v_lane2, sh_r);

        __m256i p0 = _mm256_blendv_epi8(_mm256_blendv_epi8(b0, g0, m0), r0, m1);
        __m256i p1 = _mm256_blendv_epi8(_mm256_blendv_epi8(g0, r0, m0), b0, m1);
        __m256i p2 = _mm256_blendv_epi8(_mm256_blendv_epi8(r0, b0, m0), g0, m1);

        __m256i bgr0 = _mm256_permute2x128_si256(p0, p1, 0 + 2 * 16);
        __m256i bgr1 = _mm256_permute2x128_si256(p2, p0, 0 + 3 * 16);
        __m256i bgr2 = _mm256_permute2x128_si256(p1, p2, 1 + 3 * 16);

        typename simd_type::scalar_t* mem_addr_lane_0 = reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 0);
        typename simd_type::scalar_t* mem_addr_lane_1 = reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 1);
        typename simd_type::scalar_t* mem_addr_lane_2 = reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 2);

        if constexpr (mode == fyx::simd::detail::interleave_store_mode::aligned)
        {
            _mm256_store_si256(reinterpret_cast<__m256i*>((mem_addr_lane_0)), bgr0);
            _mm256_store_si256(reinterpret_cast<__m256i*>((mem_addr_lane_1)), bgr1);
            _mm256_store_si256(reinterpret_cast<__m256i*>((mem_addr_lane_2)), bgr2);
        }
        else if constexpr (mode == fyx::simd::detail::interleave_store_mode::stream)
        {
            _mm256_stream_si256(reinterpret_cast<__m256i*>((mem_addr_lane_0)), bgr0);
            _mm256_stream_si256(reinterpret_cast<__m256i*>((mem_addr_lane_1)), bgr1);
            _mm256_stream_si256(reinterpret_cast<__m256i*>((mem_addr_lane_2)), bgr2);
        }
        else
        {
            _mm256_storeu_si256(reinterpret_cast<__m256i*>((mem_addr_lane_0)), bgr0);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>((mem_addr_lane_1)), bgr1);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>((mem_addr_lane_2)), bgr2);
        }
    }

    template<typename simd_type, fyx::simd::detail::interleave_store_mode mode> requires((sizeof(typename simd_type::scalar_t) == sizeof(std::uint8_t))
        && fyx::simd::is_256bits_simd_v<simd_type>)
        void store_interleave_impl_lane4(void* mem_addr, simd_type v0, simd_type v1, simd_type v2, simd_type v3)
    {
        __m256i v_lane0 = v0.data;
        __m256i v_lane1 = v1.data;
        __m256i v_lane2 = v2.data;
        __m256i v_lane3 = v3.data;

        __m256i bg0 = _mm256_unpacklo_epi8(v_lane0, v_lane1);
        __m256i bg1 = _mm256_unpackhi_epi8(v_lane0, v_lane1);

        __m256i ra0 = _mm256_unpacklo_epi8(v_lane2, v_lane3);
        __m256i ra1 = _mm256_unpackhi_epi8(v_lane2, v_lane3);

        __m256i bgra0_ = _mm256_unpacklo_epi16(bg0, ra0);
        __m256i bgra1_ = _mm256_unpackhi_epi16(bg0, ra0);
        __m256i bgra2_ = _mm256_unpacklo_epi16(bg1, ra1);
        __m256i bgra3_ = _mm256_unpackhi_epi16(bg1, ra1);

        __m256i bgra0 = _mm256_permute2x128_si256(bgra0_, bgra1_, 0 + 2 * 16);
        __m256i bgra2 = _mm256_permute2x128_si256(bgra0_, bgra1_, 1 + 3 * 16);
        __m256i bgra1 = _mm256_permute2x128_si256(bgra2_, bgra3_, 0 + 2 * 16);
        __m256i bgra3 = _mm256_permute2x128_si256(bgra2_, bgra3_, 1 + 3 * 16);

        typename simd_type::scalar_t* mem_addr_lane_0 = reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 0);
        typename simd_type::scalar_t* mem_addr_lane_1 = reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 1);
        typename simd_type::scalar_t* mem_addr_lane_2 = reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 2);
        typename simd_type::scalar_t* mem_addr_lane_3 = reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 3);

        if constexpr (mode == fyx::simd::detail::interleave_store_mode::aligned)
        {
            _mm256_store_si256(reinterpret_cast<__m256i*>((mem_addr_lane_0)), bgra0);
            _mm256_store_si256(reinterpret_cast<__m256i*>((mem_addr_lane_1)), bgra1);
            _mm256_store_si256(reinterpret_cast<__m256i*>((mem_addr_lane_2)), bgra2);
            _mm256_store_si256(reinterpret_cast<__m256i*>((mem_addr_lane_3)), bgra3);
        }
        else if constexpr (mode == fyx::simd::detail::interleave_store_mode::stream)
        {
            _mm256_stream_si256(reinterpret_cast<__m256i*>((mem_addr_lane_0)), bgra0);
            _mm256_stream_si256(reinterpret_cast<__m256i*>((mem_addr_lane_1)), bgra1);
            _mm256_stream_si256(reinterpret_cast<__m256i*>((mem_addr_lane_2)), bgra2);
            _mm256_stream_si256(reinterpret_cast<__m256i*>((mem_addr_lane_3)), bgra3);
        }
        else
        {
            _mm256_storeu_si256(reinterpret_cast<__m256i*>((mem_addr_lane_0)), bgra0);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>((mem_addr_lane_1)), bgra1);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>((mem_addr_lane_2)), bgra2);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>((mem_addr_lane_3)), bgra3);
        }
    }

    template<typename simd_type, fyx::simd::detail::interleave_store_mode mode> requires((sizeof(typename simd_type::scalar_t) == sizeof(std::uint16_t))
        && fyx::simd::is_256bits_simd_v<simd_type>)
        void store_interleave_impl_lane2(void* mem_addr, simd_type v0, simd_type v1)
    {
        __m256i v_lane0 = v0.data;
        __m256i v_lane1 = v1.data;

        __m256i xy_l = _mm256_unpacklo_epi16(v0.data, v1.data);
        __m256i xy_h = _mm256_unpackhi_epi16(v0.data, v1.data);

        __m256i xy0 = _mm256_permute2x128_si256(xy_l, xy_h, 0 + 2 * 16);
        __m256i xy1 = _mm256_permute2x128_si256(xy_l, xy_h, 1 + 3 * 16);

        typename simd_type::scalar_t* mem_addr_lane_0 = reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 0);
        typename simd_type::scalar_t* mem_addr_lane_1 = reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 1);

        if constexpr (mode == fyx::simd::detail::interleave_store_mode::aligned)
        {
            _mm256_store_si256(reinterpret_cast<__m256i*>((mem_addr_lane_0)), xy0);
            _mm256_store_si256(reinterpret_cast<__m256i*>((mem_addr_lane_1)), xy1);
        }
        else if constexpr (mode == fyx::simd::detail::interleave_store_mode::stream)
        {
            _mm256_stream_si256(reinterpret_cast<__m256i*>((mem_addr_lane_0)), xy0);
            _mm256_stream_si256(reinterpret_cast<__m256i*>((mem_addr_lane_1)), xy1);
        }
        else
        {
            _mm256_storeu_si256(reinterpret_cast<__m256i*>((mem_addr_lane_0)), xy0);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>((mem_addr_lane_1)), xy1);
        }
    }

    template<typename simd_type, fyx::simd::detail::interleave_store_mode mode> requires((sizeof(typename simd_type::scalar_t) == sizeof(std::uint16_t))
        && fyx::simd::is_256bits_simd_v<simd_type>)
        void store_interleave_impl_lane3(void* mem_addr, simd_type v0, simd_type v1, simd_type v2)
    {
        const __m256i sh_b = _mm256_setr_epi8(
            0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15, 4, 5, 10, 11,
            0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15, 4, 5, 10, 11);

        const __m256i sh_g = _mm256_setr_epi8(
            10, 11, 0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15, 4, 5,
            10, 11, 0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15, 4, 5);

        const __m256i sh_r = _mm256_setr_epi8(
            4, 5, 10, 11, 0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15,
            4, 5, 10, 11, 0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15);

        const __m256i m0 = _mm256_setr_epi8(
            0, 0, -1, -1, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, -1, -1,
            0, 0, 0, 0, -1, -1, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0);

        const __m256i m1 = _mm256_setr_epi8(
            0, 0, 0, 0, -1, -1, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0,
            -1, -1, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, -1, -1, 0, 0);

        __m256i v_lane0 = v0.data;
        __m256i v_lane1 = v1.data;
        __m256i v_lane2 = v2.data;

        __m256i b0 = _mm256_shuffle_epi8(v_lane0, sh_b);
        __m256i g0 = _mm256_shuffle_epi8(v_lane1, sh_g);
        __m256i r0 = _mm256_shuffle_epi8(v_lane2, sh_r);

        __m256i p0 = _mm256_blendv_epi8(_mm256_blendv_epi8(b0, g0, m0), r0, m1);
        __m256i p1 = _mm256_blendv_epi8(_mm256_blendv_epi8(g0, r0, m0), b0, m1);
        __m256i p2 = _mm256_blendv_epi8(_mm256_blendv_epi8(r0, b0, m0), g0, m1);

        __m256i bgr0 = _mm256_permute2x128_si256(p0, p2, 0 + 2 * 16);
        __m256i bgr2 = _mm256_permute2x128_si256(p0, p2, 1 + 3 * 16);

        typename simd_type::scalar_t* mem_addr_lane_0 = reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 0);
        typename simd_type::scalar_t* mem_addr_lane_1 = reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 1);
        typename simd_type::scalar_t* mem_addr_lane_2 = reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 2);

        if constexpr (mode == fyx::simd::detail::interleave_store_mode::aligned)
        {
            _mm256_store_si256(reinterpret_cast<__m256i*>((mem_addr_lane_0)), bgr0);
            _mm256_store_si256(reinterpret_cast<__m256i*>((mem_addr_lane_1)), p1);
            _mm256_store_si256(reinterpret_cast<__m256i*>((mem_addr_lane_2)), bgr2);
        }
        else if constexpr (mode == fyx::simd::detail::interleave_store_mode::stream)
        {
            _mm256_stream_si256(reinterpret_cast<__m256i*>((mem_addr_lane_0)), bgr0);
            _mm256_stream_si256(reinterpret_cast<__m256i*>((mem_addr_lane_1)), p1);
            _mm256_stream_si256(reinterpret_cast<__m256i*>((mem_addr_lane_2)), bgr2);
        }
        else
        {
            _mm256_storeu_si256(reinterpret_cast<__m256i*>((mem_addr_lane_0)), bgr0);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>((mem_addr_lane_1)), p1);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>((mem_addr_lane_2)), bgr2);
        }
    }

    template<typename simd_type, fyx::simd::detail::interleave_store_mode mode> requires((sizeof(typename simd_type::scalar_t) == sizeof(std::uint16_t))
        && fyx::simd::is_256bits_simd_v<simd_type>)
        void store_interleave_impl_lane4(void* mem_addr, simd_type v0, simd_type v1, simd_type v2, simd_type v3)
    {
        __m256i v_lane0 = v0.data;
        __m256i v_lane1 = v1.data;
        __m256i v_lane2 = v2.data;
        __m256i v_lane3 = v3.data;

        __m256i bg0 = _mm256_unpacklo_epi16(v_lane0, v_lane1);
        __m256i bg1 = _mm256_unpackhi_epi16(v_lane0, v_lane1);

        __m256i ra0 = _mm256_unpacklo_epi16(v_lane2, v_lane3);
        __m256i ra1 = _mm256_unpackhi_epi16(v_lane2, v_lane3);

        __m256i bgra0_ = _mm256_unpacklo_epi32(bg0, ra0);
        __m256i bgra1_ = _mm256_unpackhi_epi32(bg0, ra0);
        __m256i bgra2_ = _mm256_unpacklo_epi32(bg1, ra1);
        __m256i bgra3_ = _mm256_unpackhi_epi32(bg1, ra1);

        __m256i bgra0 = _mm256_permute2x128_si256(bgra0_, bgra1_, 0 + 2 * 16);
        __m256i bgra2 = _mm256_permute2x128_si256(bgra0_, bgra1_, 1 + 3 * 16);
        __m256i bgra1 = _mm256_permute2x128_si256(bgra2_, bgra3_, 0 + 2 * 16);
        __m256i bgra3 = _mm256_permute2x128_si256(bgra2_, bgra3_, 1 + 3 * 16);

        typename simd_type::scalar_t* mem_addr_lane_0 = reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 0);
        typename simd_type::scalar_t* mem_addr_lane_1 = reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 1);
        typename simd_type::scalar_t* mem_addr_lane_2 = reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 2);
        typename simd_type::scalar_t* mem_addr_lane_3 = reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 3);

        if constexpr (mode == fyx::simd::detail::interleave_store_mode::aligned)
        {
            _mm256_store_si256(reinterpret_cast<__m256i*>((mem_addr_lane_0)), bgra0);
            _mm256_store_si256(reinterpret_cast<__m256i*>((mem_addr_lane_1)), bgra1);
            _mm256_store_si256(reinterpret_cast<__m256i*>((mem_addr_lane_2)), bgra2);
            _mm256_store_si256(reinterpret_cast<__m256i*>((mem_addr_lane_3)), bgra3);
        }
        else if constexpr (mode == fyx::simd::detail::interleave_store_mode::stream)
        {
            _mm256_stream_si256(reinterpret_cast<__m256i*>((mem_addr_lane_0)), bgra0);
            _mm256_stream_si256(reinterpret_cast<__m256i*>((mem_addr_lane_1)), bgra1);
            _mm256_stream_si256(reinterpret_cast<__m256i*>((mem_addr_lane_2)), bgra2);
            _mm256_stream_si256(reinterpret_cast<__m256i*>((mem_addr_lane_3)), bgra3);
        }
        else
        {
            _mm256_storeu_si256(reinterpret_cast<__m256i*>((mem_addr_lane_0)), bgra0);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>((mem_addr_lane_1)), bgra1);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>((mem_addr_lane_2)), bgra2);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>((mem_addr_lane_3)), bgra3);
        }
    }

    template<typename simd_type, fyx::simd::detail::interleave_store_mode mode> requires((sizeof(typename simd_type::scalar_t) == sizeof(std::uint32_t))
        && fyx::simd::is_256bits_simd_v<simd_type>)
        void store_interleave_impl_lane2(void* mem_addr, simd_type v0, simd_type v1)
    {
        __m256i v_lane0 = v0.data;
        __m256i v_lane1 = v1.data;

        __m256i xy_l = _mm256_unpacklo_epi32(v_lane0, v_lane1);
        __m256i xy_h = _mm256_unpackhi_epi32(v_lane0, v_lane1);

        __m256i xy0 = _mm256_permute2x128_si256(xy_l, xy_h, 0 + 2 * 16);
        __m256i xy1 = _mm256_permute2x128_si256(xy_l, xy_h, 1 + 3 * 16);

        typename simd_type::scalar_t* mem_addr_lane_0 = reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 0);
        typename simd_type::scalar_t* mem_addr_lane_1 = reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 1);

        if constexpr (mode == fyx::simd::detail::interleave_store_mode::aligned)
        {
            _mm256_store_si256(reinterpret_cast<__m256i*>((mem_addr_lane_0)), xy0);
            _mm256_store_si256(reinterpret_cast<__m256i*>((mem_addr_lane_1)), xy1);
        }
        else if constexpr (mode == fyx::simd::detail::interleave_store_mode::stream)
        {
            _mm256_stream_si256(reinterpret_cast<__m256i*>((mem_addr_lane_0)), xy0);
            _mm256_stream_si256(reinterpret_cast<__m256i*>((mem_addr_lane_1)), xy1);
        }
        else
        {
            _mm256_storeu_si256(reinterpret_cast<__m256i*>((mem_addr_lane_0)), xy0);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>((mem_addr_lane_1)), xy1);
        }
    }

    template<typename simd_type, fyx::simd::detail::interleave_store_mode mode> requires((sizeof(typename simd_type::scalar_t) == sizeof(std::uint32_t))
        && fyx::simd::is_256bits_simd_v<simd_type>)
        void store_interleave_impl_lane3(void* mem_addr, simd_type v0, simd_type v1, simd_type v2)
    {
        __m256i v_lane0 = v0.data;
        __m256i v_lane1 = v1.data;
        __m256i v_lane2 = v2.data;

        __m256i b0 = _mm256_shuffle_epi32(v_lane0, 0x6c);
        __m256i g0 = _mm256_shuffle_epi32(v_lane1, 0xb1);
        __m256i r0 = _mm256_shuffle_epi32(v_lane2, 0xc6);

        __m256i p0 = _mm256_blend_epi32(_mm256_blend_epi32(b0, g0, 0x92), r0, 0x24);
        __m256i p1 = _mm256_blend_epi32(_mm256_blend_epi32(g0, r0, 0x92), b0, 0x24);
        __m256i p2 = _mm256_blend_epi32(_mm256_blend_epi32(r0, b0, 0x92), g0, 0x24);

        __m256i bgr0 = _mm256_permute2x128_si256(p0, p1, 0 + 2 * 16);
        __m256i bgr2 = _mm256_permute2x128_si256(p0, p1, 1 + 3 * 16);

        typename simd_type::scalar_t* mem_addr_lane_0 = reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 0);
        typename simd_type::scalar_t* mem_addr_lane_1 = reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 1);
        typename simd_type::scalar_t* mem_addr_lane_2 = reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 2);

        if constexpr (mode == fyx::simd::detail::interleave_store_mode::aligned)
        {
            _mm256_store_si256(reinterpret_cast<__m256i*>((mem_addr_lane_0)), bgr0);
            _mm256_store_si256(reinterpret_cast<__m256i*>((mem_addr_lane_1)), p2);
            _mm256_store_si256(reinterpret_cast<__m256i*>((mem_addr_lane_2)), bgr2);
        }
        else if constexpr (mode == fyx::simd::detail::interleave_store_mode::stream)
        {
            _mm256_stream_si256(reinterpret_cast<__m256i*>((mem_addr_lane_0)), bgr0);
            _mm256_stream_si256(reinterpret_cast<__m256i*>((mem_addr_lane_1)), p2);
            _mm256_stream_si256(reinterpret_cast<__m256i*>((mem_addr_lane_2)), bgr2);
        }
        else
        {
            _mm256_storeu_si256(reinterpret_cast<__m256i*>((mem_addr_lane_0)), bgr0);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>((mem_addr_lane_1)), p2);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>((mem_addr_lane_2)), bgr2);
        }
    }

    template<typename simd_type, fyx::simd::detail::interleave_store_mode mode> requires((sizeof(typename simd_type::scalar_t) == sizeof(std::uint32_t))
        && fyx::simd::is_256bits_simd_v<simd_type>)
        void store_interleave_impl_lane4(void* mem_addr, simd_type v0, simd_type v1, simd_type v2, simd_type v3)
    {
        __m256i v_lane0 = v0.data;
        __m256i v_lane1 = v1.data;
        __m256i v_lane2 = v2.data;
        __m256i v_lane3 = v3.data;

        __m256i bg0 = _mm256_unpacklo_epi32(v_lane0, v_lane1);
        __m256i bg1 = _mm256_unpackhi_epi32(v_lane0, v_lane1);

        __m256i ra0 = _mm256_unpacklo_epi32(v_lane2, v_lane3);
        __m256i ra1 = _mm256_unpackhi_epi32(v_lane2, v_lane3);

        __m256i bgra0_ = _mm256_unpacklo_epi64(bg0, ra0);
        __m256i bgra1_ = _mm256_unpackhi_epi64(bg0, ra0);
        __m256i bgra2_ = _mm256_unpacklo_epi64(bg1, ra1);
        __m256i bgra3_ = _mm256_unpackhi_epi64(bg1, ra1);

        __m256i bgra0 = _mm256_permute2x128_si256(bgra0_, bgra1_, 0 + 2 * 16);
        __m256i bgra2 = _mm256_permute2x128_si256(bgra0_, bgra1_, 1 + 3 * 16);
        __m256i bgra1 = _mm256_permute2x128_si256(bgra2_, bgra3_, 0 + 2 * 16);
        __m256i bgra3 = _mm256_permute2x128_si256(bgra2_, bgra3_, 1 + 3 * 16);

        typename simd_type::scalar_t* mem_addr_lane_0 = reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 0);
        typename simd_type::scalar_t* mem_addr_lane_1 = reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 1);
        typename simd_type::scalar_t* mem_addr_lane_2 = reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 2);
        typename simd_type::scalar_t* mem_addr_lane_3 = reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 3);

        if constexpr (mode == fyx::simd::detail::interleave_store_mode::aligned)
        {
            _mm256_store_si256(reinterpret_cast<__m256i*>((mem_addr_lane_0)), bgra0);
            _mm256_store_si256(reinterpret_cast<__m256i*>((mem_addr_lane_1)), bgra1);
            _mm256_store_si256(reinterpret_cast<__m256i*>((mem_addr_lane_2)), bgra2);
            _mm256_store_si256(reinterpret_cast<__m256i*>((mem_addr_lane_3)), bgra3);
        }
        else if constexpr (mode == fyx::simd::detail::interleave_store_mode::stream)
        {
            _mm256_stream_si256(reinterpret_cast<__m256i*>((mem_addr_lane_0)), bgra0);
            _mm256_stream_si256(reinterpret_cast<__m256i*>((mem_addr_lane_1)), bgra1);
            _mm256_stream_si256(reinterpret_cast<__m256i*>((mem_addr_lane_2)), bgra2);
            _mm256_stream_si256(reinterpret_cast<__m256i*>((mem_addr_lane_3)), bgra3);
        }
        else
        {
            _mm256_storeu_si256(reinterpret_cast<__m256i*>((mem_addr_lane_0)), bgra0);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>((mem_addr_lane_1)), bgra1);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>((mem_addr_lane_2)), bgra2);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>((mem_addr_lane_3)), bgra3);
        }
    }

    template<typename simd_type, fyx::simd::detail::interleave_store_mode mode> requires((sizeof(typename simd_type::scalar_t) == sizeof(std::uint64_t))
        && fyx::simd::is_256bits_simd_v<simd_type>)
        void store_interleave_impl_lane2(void* mem_addr, simd_type v0, simd_type v1)
    {
        __m256i v_lane0 = v0.data;
        __m256i v_lane1 = v1.data;

        __m256i xy_l = _mm256_unpacklo_epi64(v_lane0, v_lane1);
        __m256i xy_h = _mm256_unpackhi_epi64(v_lane0, v_lane1);

        __m256i xy0 = _mm256_permute2x128_si256(xy_l, xy_h, 0 + 2 * 16);
        __m256i xy1 = _mm256_permute2x128_si256(xy_l, xy_h, 1 + 3 * 16);

        typename simd_type::scalar_t* mem_addr_lane_0 = reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 0);
        typename simd_type::scalar_t* mem_addr_lane_1 = reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 1);

        if constexpr (mode == fyx::simd::detail::interleave_store_mode::aligned)
        {
            _mm256_store_si256(reinterpret_cast<__m256i*>((mem_addr_lane_0)), xy0);
            _mm256_store_si256(reinterpret_cast<__m256i*>((mem_addr_lane_1)), xy1);
        }
        else if constexpr (mode == fyx::simd::detail::interleave_store_mode::stream)
        {
            _mm256_stream_si256(reinterpret_cast<__m256i*>((mem_addr_lane_0)), xy0);
            _mm256_stream_si256(reinterpret_cast<__m256i*>((mem_addr_lane_1)), xy1);
        }
        else
        {
            _mm256_storeu_si256(reinterpret_cast<__m256i*>((mem_addr_lane_0)), xy0);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>((mem_addr_lane_1)), xy1);
        }
    }

    template<typename simd_type, fyx::simd::detail::interleave_store_mode mode> requires((sizeof(typename simd_type::scalar_t) == sizeof(std::uint64_t))
        && fyx::simd::is_256bits_simd_v<simd_type>)
        void store_interleave_impl_lane3(void* mem_addr, simd_type v0, simd_type v1, simd_type v2)
    {
        __m256i s01 = _mm256_unpacklo_epi64(v0.data, v1.data);
        __m256i s12 = _mm256_unpackhi_epi64(v1.data, v2.data);
        __m256i s20 = _mm256_blend_epi32(v2.data, v0.data, 0xcc);

        __m256i bgr0 = _mm256_permute2x128_si256(s01, s20, 0 + 2 * 16);
        __m256i bgr1 = _mm256_blend_epi32(s01, s12, 0x0f);
        __m256i bgr2 = _mm256_permute2x128_si256(s20, s12, 1 + 3 * 16);

        typename simd_type::scalar_t* mem_addr_lane_0 = reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 0);
        typename simd_type::scalar_t* mem_addr_lane_1 = reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 1);
        typename simd_type::scalar_t* mem_addr_lane_2 = reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 2);

        if constexpr (mode == fyx::simd::detail::interleave_store_mode::aligned)
        {
            _mm256_store_si256(reinterpret_cast<__m256i*>((mem_addr_lane_0)), bgr0);
            _mm256_store_si256(reinterpret_cast<__m256i*>((mem_addr_lane_1)), bgr1);
            _mm256_store_si256(reinterpret_cast<__m256i*>((mem_addr_lane_2)), bgr2);
        }
        else if constexpr (mode == fyx::simd::detail::interleave_store_mode::stream)
        {
            _mm256_stream_si256(reinterpret_cast<__m256i*>((mem_addr_lane_0)), bgr0);
            _mm256_stream_si256(reinterpret_cast<__m256i*>((mem_addr_lane_1)), bgr1);
            _mm256_stream_si256(reinterpret_cast<__m256i*>((mem_addr_lane_2)), bgr2);
        }
        else
        {
            _mm256_storeu_si256(reinterpret_cast<__m256i*>((mem_addr_lane_0)), bgr0);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>((mem_addr_lane_1)), bgr1);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>((mem_addr_lane_2)), bgr2);
        }
    }

    template<typename simd_type, fyx::simd::detail::interleave_store_mode mode> requires((sizeof(typename simd_type::scalar_t) == sizeof(std::uint64_t))
        && fyx::simd::is_256bits_simd_v<simd_type>)
        void store_interleave_impl_lane4(void* mem_addr, simd_type v0, simd_type v1, simd_type v2, simd_type v3)
    {
        __m256i bg0 = _mm256_unpacklo_epi64(v0.data, v1.data);
        __m256i bg1 = _mm256_unpackhi_epi64(v0.data, v1.data);

        __m256i ra0 = _mm256_unpacklo_epi64(v2.data, v3.data);
        __m256i ra1 = _mm256_unpackhi_epi64(v2.data, v3.data);

        __m256i bgra0 = _mm256_permute2x128_si256(bg0, ra0, 0 + 2 * 16);
        __m256i bgra1 = _mm256_permute2x128_si256(bg1, ra1, 0 + 2 * 16);
        __m256i bgra2 = _mm256_permute2x128_si256(bg0, ra0, 1 + 3 * 16);
        __m256i bgra3 = _mm256_permute2x128_si256(bg1, ra1, 1 + 3 * 16);

        typename simd_type::scalar_t* mem_addr_lane_0 = reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 0);
        typename simd_type::scalar_t* mem_addr_lane_1 = reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 1);
        typename simd_type::scalar_t* mem_addr_lane_2 = reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 2);
        typename simd_type::scalar_t* mem_addr_lane_3 = reinterpret_cast<typename simd_type::scalar_t*>(mem_addr) + (simd_type::lane_width * 3);

        if constexpr (mode == fyx::simd::detail::interleave_store_mode::aligned)
        {
            _mm256_store_si256(reinterpret_cast<__m256i*>((mem_addr_lane_0)), bgra0);
            _mm256_store_si256(reinterpret_cast<__m256i*>((mem_addr_lane_1)), bgra1);
            _mm256_store_si256(reinterpret_cast<__m256i*>((mem_addr_lane_2)), bgra2);
            _mm256_store_si256(reinterpret_cast<__m256i*>((mem_addr_lane_3)), bgra3);
        }
        else if constexpr (mode == fyx::simd::detail::interleave_store_mode::stream)
        {
            _mm256_stream_si256(reinterpret_cast<__m256i*>((mem_addr_lane_0)), bgra0);
            _mm256_stream_si256(reinterpret_cast<__m256i*>((mem_addr_lane_1)), bgra1);
            _mm256_stream_si256(reinterpret_cast<__m256i*>((mem_addr_lane_2)), bgra2);
            _mm256_stream_si256(reinterpret_cast<__m256i*>((mem_addr_lane_3)), bgra3);
        }
        else
        {
            _mm256_storeu_si256(reinterpret_cast<__m256i*>((mem_addr_lane_0)), bgra0);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>((mem_addr_lane_1)), bgra1);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>((mem_addr_lane_2)), bgra2);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>((mem_addr_lane_3)), bgra3);
        }
    }


    template<fyx::simd::detail::interleave_store_mode store_mode, typename ... simd_type>
    void store_interleave_(void* mem_addr, simd_type&& ... args)
    {
        constexpr std::size_t count_lane = sizeof...(simd_type);
        static_assert(count_lane > 0, "At least one simd vector is required");

        using first_dereferenced_type = std::decay_t<std::tuple_element_t<0, std::tuple<simd_type...>>>;
        using scalar_t = typename first_dereferenced_type::scalar_t;

        static_assert(((is_basic_simd_v<std::decay_t<simd_type>> &&
            std::is_same_v<typename std::decay_t<simd_type>::scalar_t, scalar_t> &&
            std::decay_t<simd_type>::lane_width == first_dereferenced_type::lane_width) && ...),
            "All simd types must have the same scalar type and lane width");

        if constexpr (count_lane == 2)
        {
            return fyx::simd::detail::store_interleave_impl_lane2<
                first_dereferenced_type,
                store_mode>(mem_addr, args ...);
        }
        else if constexpr (count_lane == 3)
        {
            return fyx::simd::detail::store_interleave_impl_lane3<
                first_dereferenced_type,
                store_mode>(mem_addr, args ...);
        }
        else if constexpr (count_lane == 4)
        {
            return fyx::simd::detail::store_interleave_impl_lane4<
                first_dereferenced_type,
                store_mode>(mem_addr, args ...);
        }
        else
        {
            static_assert(sizeof(scalar_t) * count_lane * first_dereferenced_type::lane_width <= 512 * 1024,
                "The total size of nested_array exceeds 512KB, which may cause stack overflow. ");

            constexpr std::size_t lane_width = first_dereferenced_type::lane_width;
            alignas(alignof(typename first_dereferenced_type::vector_t)) scalar_t nested_array[count_lane][lane_width] = {};

            [&] <std::size_t... I>(std::index_sequence<I...>)
            {
                auto vectors_tuple = std::forward_as_tuple(args...);
                ((fyx::simd::store_aligned(std::get<I>(vectors_tuple).data, nested_array[I])), ...);
            }(std::make_index_sequence<count_lane>{});

            scalar_t* dst = reinterpret_cast<scalar_t*>(mem_addr);
            for (std::size_t elem_idx = 0; elem_idx < lane_width; ++elem_idx)
            {
                for (std::size_t vec_idx = 0; vec_idx < count_lane; ++vec_idx)
                {
                    dst[elem_idx * count_lane + vec_idx] = nested_array[vec_idx][elem_idx];
                }
            }

            if constexpr (store_mode == fyx::simd::detail::interleave_store_mode::stream)
            {
                std::atomic_thread_fence(std::memory_order_release);
            }
        }
    }

    template<fyx::simd::detail::interleave_load_mode load_mode, typename ... addr_of_simd_type>
    void load_interleave_(const void* src_ptr, addr_of_simd_type&& ... vsrc)
    {
        constexpr std::size_t count_lane = sizeof...(addr_of_simd_type);
        static_assert(count_lane > 1, "At least two simd vector is required");

        using first_dereferenced_type = std::remove_pointer_t<std::decay_t<std::tuple_element_t<0, std::tuple<addr_of_simd_type...>>>>;
        using scalar_t = typename first_dereferenced_type::scalar_t;
        using vector_t = typename first_dereferenced_type::vector_t;

        static_assert(((std::is_pointer_v<std::decay_t<addr_of_simd_type>> &&
            std::is_same_v<std::remove_pointer_t<std::decay_t<addr_of_simd_type>>, first_dereferenced_type>) && ...),
            "All addr_of_simd_type arguments must be pointers to the same basic_simd specialization");

        static_assert(is_basic_simd_v<first_dereferenced_type>, "Pointed type must be a basic_simd specialization");

        static_assert(((is_basic_simd_v<std::remove_pointer_t<std::decay_t<addr_of_simd_type>>> &&
            std::is_same_v<typename std::remove_pointer_t<std::decay_t<addr_of_simd_type>>::scalar_t, scalar_t> &&
            std::remove_pointer_t<std::decay_t<addr_of_simd_type>>::lane_width == first_dereferenced_type::lane_width) && ...),
            "All pointed simd types must have the same scalar type and lane width");

        const scalar_t* mem_addr = reinterpret_cast<const scalar_t*>(src_ptr);

        if constexpr (count_lane == 2)
        {
            fyx::simd::detail::load_interleave_impl_lane2<first_dereferenced_type, load_mode>(
                mem_addr, std::forward<addr_of_simd_type>(vsrc) ...);
        }
        else if constexpr (count_lane == 3)
        {
            fyx::simd::detail::load_interleave_impl_lane3<first_dereferenced_type, load_mode>(
                mem_addr, std::forward<addr_of_simd_type>(vsrc) ...);
        }
        else if constexpr (count_lane == 4)
        {
            fyx::simd::detail::load_interleave_impl_lane4<first_dereferenced_type, load_mode>(
                mem_addr, std::forward<addr_of_simd_type>(vsrc) ...);
        }
        else
        {
            constexpr std::size_t lane_width = first_dereferenced_type::lane_width;
            static_assert(sizeof(scalar_t) * count_lane * lane_width <= 512 * 1024,
                "The total size of nested_array exceeds 512KB, which may cause stack overflow. ");

            alignas(alignof(vector_t)) scalar_t temp_arrays[count_lane][lane_width];
            for (std::size_t i = 0; i < lane_width; ++i)
            {
                std::size_t base_index = i * count_lane;
                [&] <std::size_t... J>(std::index_sequence<J...>)
                {
                    ((temp_arrays[J][i] = mem_addr[base_index + J]), ...);
                }(std::make_index_sequence<count_lane>{});
            }

            [&] <std::size_t... J>(std::index_sequence<J...>)
            {
                auto pointers = std::forward_as_tuple(vsrc...);
                ((*std::get<J>(pointers) = fyx::simd::load_aligned<first_dereferenced_type>(temp_arrays[J])), ...);
            }(std::make_index_sequence<count_lane>{});
        }
    }
}

namespace fyx::simd
{
    template<typename ... simd_type> requires((fyx::simd::is_basic_simd_v<std::remove_cvref_t<simd_type>> && ...))
    void store_interleave_unaligned(void* mem_addr, simd_type&& ... args)
    {
        return fyx::simd::detail::store_interleave_<fyx::simd::detail::interleave_store_mode::unaligned>(
            mem_addr, std::forward<simd_type>(args)...);
    }

    template<typename ... simd_type> requires((fyx::simd::is_basic_simd_v<std::remove_cvref_t<simd_type>> && ...))
    void store_interleave_aligned(void* mem_addr, simd_type&& ... args)
    {
        return fyx::simd::detail::store_interleave_<fyx::simd::detail::interleave_store_mode::aligned>(
            mem_addr, std::forward<simd_type>(args)...);
    }

    template<typename ... simd_type> requires((fyx::simd::is_basic_simd_v<std::remove_cvref_t<simd_type>> && ...))
    void store_interleave_stream(void* mem_addr, simd_type&& ... args)
    {
        return fyx::simd::detail::store_interleave_<fyx::simd::detail::interleave_store_mode::stream>(
            mem_addr, std::forward<simd_type>(args)...);
    }

    template<typename ... addr_of_simd_type> requires (fyx::simd::is_basic_simd_v<
        std::remove_cv_t<std::remove_pointer_t<std::remove_reference_t<addr_of_simd_type>>>> && ...)
    void load_interleave_unaligned(const void* src_ptr, addr_of_simd_type&& ... vsrc)
    {
        fyx::simd::detail::load_interleave_<detail::interleave_load_mode::unaligned>(
            src_ptr, std::forward<addr_of_simd_type>(vsrc)...);
    }

    template<typename ... addr_of_simd_type> requires (fyx::simd::is_basic_simd_v<
        std::remove_cv_t<std::remove_pointer_t<std::remove_reference_t<addr_of_simd_type>>>> && ...)
    void load_interleave_aligned(const void* src_ptr, addr_of_simd_type&& ... vsrc)
    {
        fyx::simd::detail::load_interleave_<detail::interleave_load_mode::aligned>(
            src_ptr, std::forward<addr_of_simd_type>(vsrc)...);
    }


    uint8x16 interleave_split_even(uint8x32 input)
    {
        __m256i vsrc = input.data;
        const __m128i even_mask = _mm_set_epi8(
            0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
            14, 12, 10, 8, 6, 4, 2, 0
        );

        return uint8x16{ _mm_unpacklo_epi64(
            _mm_shuffle_epi8(detail::split_low(vsrc), even_mask),
            _mm_shuffle_epi8(detail::split_high(vsrc), even_mask)) };
    }

    uint16x8 interleave_split_even(uint16x16 input)
    {
        __m256i vsrc = input.data;
        const __m128i even_mask = _mm_set_epi8(
            0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
            13, 12, 9, 8, 5, 4, 1, 0
        );

        return uint16x8{ _mm_unpacklo_epi64(
            _mm_shuffle_epi8(detail::split_low(vsrc), even_mask),
            _mm_shuffle_epi8(detail::split_high(vsrc), even_mask)) };
    }

    uint32x4 interleave_split_even(uint32x8 input)
    {
        __m256i vsrc = input.data;
        return uint32x4{ _mm_unpacklo_epi64(
            _mm_shuffle_epi32(detail::split_low(vsrc), _MM_SHUFFLE(2, 0, 2, 0)),
            _mm_shuffle_epi32(detail::split_high(vsrc), _MM_SHUFFLE(2, 0, 2, 0))) };
    }

    uint64x2 interleave_split_even(uint64x4 input)
    {
        __m256i vsrc = input.data;
        __m256i permuted = _mm256_permute4x64_epi64(vsrc, 0x08);
        __m128i vdst = _mm256_castsi256_si128(permuted);
        return uint64x2{ vdst };
    }

    sint8x16 interleave_split_even(sint8x32 input) 
    { return sint8x16{ fyx::simd::interleave_split_even(uint8x32{ input.data }) }; }

    sint16x8 interleave_split_even(sint16x16 input) 
    { return sint16x8{ fyx::simd::interleave_split_even(uint16x16{ input.data }) }; }

    sint32x4 interleave_split_even(sint32x8 input) 
    { return sint32x4{ fyx::simd::interleave_split_even(uint32x8{ input.data }) }; }

    sint64x2 interleave_split_even(sint64x4 input) 
    { return sint64x2{ fyx::simd::interleave_split_even(uint64x4{ input.data }) }; }

    float32x4 interleave_split_even(float32x8 input)
    {
        __m256 vsrc = input.data;
        __m128 low = detail::split_low(vsrc);
        __m128 high = detail::split_high(vsrc);
        return float32x4{ _mm_shuffle_ps(low, high, _MM_SHUFFLE(2, 0, 2, 0)) };
    }

    float64x2 interleave_split_even(float64x4 input)
    {
        __m256d vsrc = input.data;
        __m256d permuted = _mm256_permute4x64_pd(vsrc, 0x08);
        __m128d vdst = _mm256_castpd256_pd128(permuted);
        return float64x2{ vdst };
    }
#if defined(_FOYE_SIMD_HAS_FP16_)
    float16x8 interleave_even(float16x16 input) 
    { return float16x8{ fyx::simd::interleave_split_even(uint16x16{ input.data }) }; }
#endif
#if defined(_FOYE_SIMD_HAS_BF16_)
    bfloat16x8 interleave_even(bfloat16x16 input) 
    { return bfloat16x8{ fyx::simd::interleave_split_even(uint16x16{ input.data }) }; }
#endif


    uint8x16 interleave_split_odd(uint8x32 input)
    {
        __m256i vsrc = input.data;
        const __m128i odd_mask = _mm_set_epi8(
            0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
            15, 13, 11, 9, 7, 5, 3, 1
        );

        return uint8x16{ _mm_unpacklo_epi64(
            _mm_shuffle_epi8(detail::split_low(vsrc), odd_mask),
            _mm_shuffle_epi8(detail::split_high(vsrc), odd_mask)) };
    }

    uint16x8 interleave_split_odd(uint16x16 input)
    {
        __m256i vsrc = input.data;
        const __m128i odd_mask = _mm_set_epi8(
            0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
            15, 14, 11, 10, 7, 6, 3, 2
        );

        return uint16x8{ _mm_unpacklo_epi64(
            _mm_shuffle_epi8(detail::split_low(vsrc), odd_mask),
            _mm_shuffle_epi8(detail::split_high(vsrc), odd_mask)) };
    }

    uint32x4 interleave_split_odd(uint32x8 input)
    {
        __m256i vsrc = input.data;
        return uint32x4{ _mm_unpacklo_epi64(
            _mm_shuffle_epi32(detail::split_low(vsrc), _MM_SHUFFLE(3, 1, 3, 1)),
            _mm_shuffle_epi32(detail::split_high(vsrc), _MM_SHUFFLE(3, 1, 3, 1))) };
    }

    uint64x2 interleave_split_odd(uint64x4 input)
    {
        __m256i vsrc = input.data;
        __m256i permuted = _mm256_permute4x64_epi64(vsrc, 0x0D);
        __m128i vdst = _mm256_castsi256_si128(permuted);
        return uint64x2{ vdst };
    }

    sint8x16 interleave_split_odd(sint8x32 input) 
    { return sint8x16{ fyx::simd::interleave_split_odd(uint8x32{ input.data }) }; }

    sint16x8 interleave_split_odd(sint16x16 input)
    { return sint16x8{ fyx::simd::interleave_split_odd(uint16x16{ input.data }) }; }

    sint32x4 interleave_split_odd(sint32x8 input) 
    { return sint32x4{ fyx::simd::interleave_split_odd(uint32x8{ input.data }) }; }

    sint64x2 interleave_split_odd(sint64x4 input) 
    { return sint64x2{ fyx::simd::interleave_split_odd(uint64x4{ input.data }) }; }

    float32x4 interleave_split_odd(float32x8 input)
    {
        __m256 vsrc = input.data;
        return float32x4{ _mm_shuffle_ps(
            detail::split_low(vsrc),
            detail::split_high(vsrc),
            _MM_SHUFFLE(3, 1, 3, 1)) };
    }

    float64x2 interleave_split_odd(float64x4 input)
    {
        __m256d vsrc = input.data;
        __m256d permuted = _mm256_permute4x64_pd(vsrc, 0x0D);
        __m128d vdst = _mm256_castpd256_pd128(permuted);
        return float64x2{ vdst };
    }

#if defined(_FOYE_SIMD_HAS_FP16_)
    float16x8 interleave_split_odd(float16x16 input) 
    { return float16x8{ fyx::simd::interleave_split_odd(uint16x16{ input.data }) }; }
#endif
#if defined(_FOYE_SIMD_HAS_BF16_)
    bfloat16x8 interleave_split_odd(bfloat16x16 input) 
    { return bfloat16x8{ fyx::simd::interleave_split_odd(uint16x16{ input.data }) }; }
#endif

    uint8x32 interleave_concat(uint8x16 odd_begin, uint8x16 src2)
    {
        __m128i a = odd_begin.data;
        __m128i b = src2.data;
        __m128i low = _mm_unpacklo_epi8(a, b);
        __m128i high = _mm_unpackhi_epi8(a, b);
        __m256i result = detail::merge(low, high);
        return uint8x32{ result };
    }

    uint16x16 interleave_concat(uint16x8 odd_begin, uint16x8 src2)
    {
        __m128i a = odd_begin.data;
        __m128i b = src2.data;
        __m128i low = _mm_unpacklo_epi16(a, b);
        __m128i high = _mm_unpackhi_epi16(a, b);
        __m256i result = detail::merge(low, high);
        return uint16x16{ result };
    }

    uint32x8 interleave_concat(uint32x4 odd_begin, uint32x4 src2)
    {
        __m128i a = odd_begin.data;
        __m128i b = src2.data;
        __m128i low = _mm_unpacklo_epi32(a, b);
        __m128i high = _mm_unpackhi_epi32(a, b);
        __m256i result = detail::merge(low, high);
        return uint32x8{ result };
    }

    uint64x4 interleave_concat(uint64x2 odd_begin, uint64x2 src2)
    {
        __m128i a = odd_begin.data;
        __m128i b = src2.data;
        __m128i low = _mm_unpacklo_epi64(a, b);
        __m128i high = _mm_unpackhi_epi64(a, b);
        __m256i result = detail::merge(low, high);
        return uint64x4{ result };
    }

    sint8x32 interleave_concat(sint8x16 odd_begin, sint8x16 src2)
    {
        return fyx::simd::reinterpret<sint8x32>(
            fyx::simd::interleave_concat(
                fyx::simd::reinterpret<uint8x16>(odd_begin),
                fyx::simd::reinterpret<uint8x16>(src2)));
    }

    sint16x16 interleave_concat(sint16x8 odd_begin, sint16x8 src2)
    {
        return fyx::simd::reinterpret<sint16x16>(
            fyx::simd::interleave_concat(
                fyx::simd::reinterpret<uint16x8>(odd_begin),
                fyx::simd::reinterpret<uint16x8>(src2)));
    }

    sint32x8 interleave_concat(sint32x4 odd_begin, sint32x4 src2)
    {
        return fyx::simd::reinterpret<sint32x8>(
            fyx::simd::interleave_concat(
                fyx::simd::reinterpret<uint32x4>(odd_begin),
                fyx::simd::reinterpret<uint32x4>(src2)));
    }

    sint64x4 interleave_concat(sint64x2 odd_begin, sint64x2 src2)
    {
        return fyx::simd::reinterpret<sint64x4>(
            fyx::simd::interleave_concat(
                fyx::simd::reinterpret<uint64x2>(odd_begin),
                fyx::simd::reinterpret<uint64x2>(src2)));
    }

    float32x8 interleave_concat(float32x4 odd_begin, float32x4 src2)
    {
        __m128 v_oddbegin = odd_begin.data;
        __m128 v_src2 = src2.data;

        __m128 low128 = _mm_unpacklo_ps(v_oddbegin, v_src2);
        __m128 high128 = _mm_unpackhi_ps(v_oddbegin, v_src2);

        __m256 result = detail::merge(low128, high128);
        return float32x8(result);
    }

    float64x4 interleave_concat(float64x2 odd_begin, float64x2 src2)
    {
        __m128d v_oddbegin = odd_begin.data;
        __m128d v_src2 = src2.data;

        __m128d low128 = _mm_unpacklo_pd(v_oddbegin, v_src2);
        __m128d high128 = _mm_unpackhi_pd(v_oddbegin, v_src2);

        __m256d result = detail::merge(low128, high128);
        return float64x4(result);
    }

#if defined(_FOYE_SIMD_HAS_FP16_)
    float16x16 interleave_concat(float16x8 odd_begin, float16x8 src2)
    {
        return fyx::simd::reinterpret<float16x16>(
            fyx::simd::interleave_concat(
                fyx::simd::reinterpret<uint16x8>(odd_begin),
                fyx::simd::reinterpret<uint16x8>(src2)));
    }
#endif
#if defined(_FOYE_SIMD_HAS_BF16_)
    bfloat16x16 interleave_concat(bfloat16x8 odd_begin, bfloat16x8 src2)
    {
        return fyx::simd::reinterpret<bfloat16x16>(
            fyx::simd::interleave_concat(
                fyx::simd::reinterpret<uint16x8>(odd_begin),
                fyx::simd::reinterpret<uint16x8>(src2)));
    }
#endif



    uint8x16 interleave_concat_low(uint8x16 odd_begin, uint8x16 src2) { return uint8x16{ _mm_unpacklo_epi8(odd_begin.data, src2.data) }; }
    uint16x8 interleave_concat_low(uint16x8 odd_begin, uint16x8 src2) { return uint16x8{ _mm_unpacklo_epi16(odd_begin.data, src2.data) }; }
    uint32x4 interleave_concat_low(uint32x4 odd_begin, uint32x4 src2) { return uint32x4{ _mm_unpacklo_epi32(odd_begin.data, src2.data) }; }
    uint64x2 interleave_concat_low(uint64x2 odd_begin, uint64x2 src2) { return uint64x2{ _mm_unpacklo_epi64(odd_begin.data, src2.data) }; }
    sint8x16 interleave_concat_low(sint8x16 odd_begin, sint8x16 src2) { return sint8x16{ _mm_unpacklo_epi8(odd_begin.data, src2.data) }; }
    sint16x8 interleave_concat_low(sint16x8 odd_begin, sint16x8 src2) { return sint16x8{ _mm_unpacklo_epi16(odd_begin.data, src2.data) }; }
    sint32x4 interleave_concat_low(sint32x4 odd_begin, sint32x4 src2) { return sint32x4{ _mm_unpacklo_epi32(odd_begin.data, src2.data) }; }
    sint64x2 interleave_concat_low(sint64x2 odd_begin, sint64x2 src2) { return sint64x2{ _mm_unpacklo_epi64(odd_begin.data, src2.data) }; }
    float32x4 interleave_concat_low(float32x4 odd_begin, float32x4 src2) { return float32x4{ _mm_unpacklo_ps(odd_begin.data, src2.data) }; }
    float64x2 interleave_concat_low(float64x2 odd_begin, float64x2 src2) { return float64x2{ _mm_unpacklo_pd(odd_begin.data, src2.data) }; }

    uint8x16 interleave_concat_high(uint8x16 odd_begin, uint8x16 src2) { return uint8x16{ _mm_unpackhi_epi8(odd_begin.data, src2.data) }; }
    uint16x8 interleave_concat_high(uint16x8 odd_begin, uint16x8 src2) { return uint16x8{ _mm_unpackhi_epi16(odd_begin.data, src2.data) }; }
    uint32x4 interleave_concat_high(uint32x4 odd_begin, uint32x4 src2) { return uint32x4{ _mm_unpackhi_epi32(odd_begin.data, src2.data) }; }
    uint64x2 interleave_concat_high(uint64x2 odd_begin, uint64x2 src2) { return uint64x2{ _mm_unpackhi_epi64(odd_begin.data, src2.data) }; }
    sint8x16 interleave_concat_high(sint8x16 odd_begin, sint8x16 src2) { return sint8x16{ _mm_unpackhi_epi8(odd_begin.data, src2.data) }; }
    sint16x8 interleave_concat_high(sint16x8 odd_begin, sint16x8 src2) { return sint16x8{ _mm_unpackhi_epi16(odd_begin.data, src2.data) }; }
    sint32x4 interleave_concat_high(sint32x4 odd_begin, sint32x4 src2) { return sint32x4{ _mm_unpackhi_epi32(odd_begin.data, src2.data) }; }
    sint64x2 interleave_concat_high(sint64x2 odd_begin, sint64x2 src2) { return sint64x2{ _mm_unpackhi_epi64(odd_begin.data, src2.data) }; }
    float32x4 interleave_concat_high(float32x4 odd_begin, float32x4 src2) { return float32x4{ _mm_unpackhi_ps(odd_begin.data, src2.data) }; }
    float64x2 interleave_concat_high(float64x2 odd_begin, float64x2 src2) { return float64x2{ _mm_unpackhi_pd(odd_begin.data, src2.data) }; }
    
    uint8x32 interleave_concat_low_per_half(uint8x32 odd_begin, uint8x32 src2) { return uint8x32{ _mm256_unpacklo_epi8(odd_begin.data, src2.data) }; }
    uint16x16 interleave_concat_low_per_half(uint16x16 odd_begin, uint16x16 src2) { return uint16x16{ _mm256_unpacklo_epi16(odd_begin.data, src2.data) }; }
    uint32x8 interleave_concat_low_per_half(uint32x8 odd_begin, uint32x8 src2) { return uint32x8{ _mm256_unpacklo_epi32(odd_begin.data, src2.data) }; }
    uint64x4 interleave_concat_low_per_half(uint64x4 odd_begin, uint64x4 src2) { return uint64x4{ _mm256_unpacklo_epi64(odd_begin.data, src2.data) }; }
    sint8x32 interleave_concat_low_per_half(sint8x32 odd_begin, sint8x32 src2) { return sint8x32{ _mm256_unpacklo_epi8(odd_begin.data, src2.data) }; }
    sint16x16 interleave_concat_low_per_half(sint16x16 odd_begin, sint16x16 src2) { return sint16x16{ _mm256_unpacklo_epi16(odd_begin.data, src2.data) }; }
    sint32x8 interleave_concat_low_per_half(sint32x8 odd_begin, sint32x8 src2) { return sint32x8{ _mm256_unpacklo_epi32(odd_begin.data, src2.data) }; }
    sint64x4 interleave_concat_low_per_half(sint64x4 odd_begin, sint64x4 src2) { return sint64x4{ _mm256_unpacklo_epi64(odd_begin.data, src2.data) }; }
    float32x8 interleave_concat_low_per_half(float32x8 odd_begin, float32x8 src2) { return float32x8{ _mm256_unpacklo_ps(odd_begin.data, src2.data) }; }
    float64x4 interleave_concat_low_per_half(float64x4 odd_begin, float64x4 src2) { return float64x4{ _mm256_unpacklo_pd(odd_begin.data, src2.data) }; }

    uint8x32 interleave_concat_high_per_half(uint8x32 odd_begin, uint8x32 src2) { return uint8x32{ _mm256_unpackhi_epi8(odd_begin.data, src2.data) }; }
    uint16x16 interleave_concat_high_per_half(uint16x16 odd_begin, uint16x16 src2) { return uint16x16{ _mm256_unpackhi_epi16(odd_begin.data, src2.data) }; }
    uint32x8 interleave_concat_high_per_half(uint32x8 odd_begin, uint32x8 src2) { return uint32x8{ _mm256_unpackhi_epi32(odd_begin.data, src2.data) }; }
    uint64x4 interleave_concat_high_per_half(uint64x4 odd_begin, uint64x4 src2) { return uint64x4{ _mm256_unpackhi_epi64(odd_begin.data, src2.data) }; }
    sint8x32 interleave_concat_high_per_half(sint8x32 odd_begin, sint8x32 src2) { return sint8x32{ _mm256_unpackhi_epi8(odd_begin.data, src2.data) }; }
    sint16x16 interleave_concat_high_per_half(sint16x16 odd_begin, sint16x16 src2) { return sint16x16{ _mm256_unpackhi_epi16(odd_begin.data, src2.data) }; }
    sint32x8 interleave_concat_high_per_half(sint32x8 odd_begin, sint32x8 src2) { return sint32x8{ _mm256_unpackhi_epi32(odd_begin.data, src2.data) }; }
    sint64x4 interleave_concat_high_per_half(sint64x4 odd_begin, sint64x4 src2) { return sint64x4{ _mm256_unpackhi_epi64(odd_begin.data, src2.data) }; }
    float32x8 interleave_concat_high_per_half(float32x8 odd_begin, float32x8 src2) { return float32x8{ _mm256_unpackhi_ps(odd_begin.data, src2.data) }; }
    float64x4 interleave_concat_high_per_half(float64x4 odd_begin, float64x4 src2) { return float64x4{ _mm256_unpackhi_pd(odd_begin.data, src2.data) }; }

#if defined(_FOYE_SIMD_HAS_FP16_)
    float16x8 interleave_concat_low(float16x8 odd_begin, float16x8 src2) 
    { return reinterpret<float16x8>(interleave_concat_low(reinterpret<uint16x8>(odd_begin), reinterpret<uint16x8>(src2))); }
    float16x8 interleave_concat_high(float16x8 odd_begin, float16x8 src2) 
    { return reinterpret<float16x8>(interleave_concat_high(reinterpret<uint16x8>(odd_begin), reinterpret<uint16x8>(src2))); }
    float16x16 interleave_concat_low_per_half(float16x16 odd_begin, float16x16 src2) 
    { return reinterpret<float16x16>(interleave_concat_low_per_half(reinterpret<uint16x16>(odd_begin), reinterpret<uint16x16>(src2))); }
    float16x16 interleave_concat_high_per_half(float16x16 odd_begin, float16x16 src2) 
    { return reinterpret<float16x16>(interleave_concat_high_per_half(reinterpret<uint16x16>(odd_begin), reinterpret<uint16x16>(src2))); }

#endif
#if defined(_FOYE_SIMD_HAS_BF16_)
    bfloat16x8 interleave_concat_low(bfloat16x8 odd_begin, bfloat16x8 src2) 
    { return reinterpret<bfloat16x8>(interleave_concat_low(reinterpret<uint16x8>(odd_begin), reinterpret<uint16x8>(src2))); }
    bfloat16x8 interleave_concat_high(bfloat16x8 odd_begin, bfloat16x8 src2) 
    { return reinterpret<bfloat16x8>(interleave_concat_high(reinterpret<uint16x8>(odd_begin), reinterpret<uint16x8>(src2))); }
    bfloat16x16 interleave_concat_low_per_half(bfloat16x16 odd_begin, bfloat16x16 src2) 
    { return reinterpret<bfloat16x16>(interleave_concat_low_per_half(reinterpret<uint16x16>(odd_begin), reinterpret<uint16x16>(src2))); }
    bfloat16x16 interleave_concat_high_per_half(bfloat16x16 odd_begin, bfloat16x16 src2) 
    { return reinterpret<bfloat16x16>(interleave_concat_high_per_half(reinterpret<uint16x16>(odd_begin), reinterpret<uint16x16>(src2))); }
#endif


#if defined(_FOYE_SIMD_ENABLE_EMULATED_)
    uint8x32 interleave_concat_low(uint8x32 odd_begin, uint8x32 src2) { return interleave_concat(odd_begin.low_part(), src2.low_part()); }
    uint16x16 interleave_concat_low(uint16x16 odd_begin, uint16x16 src2)  { return interleave_concat(odd_begin.low_part(), src2.low_part()); }
    uint32x8 interleave_concat_low(uint32x8 odd_begin, uint32x8 src2) { return interleave_concat(odd_begin.low_part(), src2.low_part()); }
    uint64x4 interleave_concat_low(uint64x4 odd_begin, uint64x4 src2) { return interleave_concat(odd_begin.low_part(), src2.low_part()); }
    sint8x32 interleave_concat_low(sint8x32 odd_begin, sint8x32 src2) { return interleave_concat(odd_begin.low_part(), src2.low_part()); }
    sint16x16 interleave_concat_low(sint16x16 odd_begin, sint16x16 src2) { return interleave_concat(odd_begin.low_part(), src2.low_part()); }
    sint32x8 interleave_concat_low(sint32x8 odd_begin, sint32x8 src2) { return interleave_concat(odd_begin.low_part(), src2.low_part()); }
    sint64x4 interleave_concat_low(sint64x4 odd_begin, sint64x4 src2) { return interleave_concat(odd_begin.low_part(), src2.low_part()); }
    float32x8 interleave_concat_low(float32x8 odd_begin, float32x8 src2) { return interleave_concat(odd_begin.low_part(), src2.low_part()); }
    float64x4 interleave_concat_low(float64x4 odd_begin, float64x4 src2) { return interleave_concat(odd_begin.low_part(), src2.low_part()); }

    uint8x32 interleave_concat_high(uint8x32 odd_begin, uint8x32 src2) { return interleave_concat(odd_begin.high_part(), src2.high_part()); }
    uint16x16 interleave_concat_high(uint16x16 odd_begin, uint16x16 src2) { return interleave_concat(odd_begin.high_part(), src2.high_part()); }
    uint32x8 interleave_concat_high(uint32x8 odd_begin, uint32x8 src2) { return interleave_concat(odd_begin.high_part(), src2.high_part()); }
    uint64x4 interleave_concat_high(uint64x4 odd_begin, uint64x4 src2) { return interleave_concat(odd_begin.high_part(), src2.high_part()); }
    sint8x32 interleave_concat_high(sint8x32 odd_begin, sint8x32 src2) { return interleave_concat(odd_begin.high_part(), src2.high_part()); }
    sint16x16 interleave_concat_high(sint16x16 odd_begin, sint16x16 src2) { return interleave_concat(odd_begin.high_part(), src2.high_part()); }
    sint32x8 interleave_concat_high(sint32x8 odd_begin, sint32x8 src2) { return interleave_concat(odd_begin.high_part(), src2.high_part()); }
    sint64x4 interleave_concat_high(sint64x4 odd_begin, sint64x4 src2) { return interleave_concat(odd_begin.high_part(), src2.high_part()); }
    float32x8 interleave_concat_high(float32x8 odd_begin, float32x8 src2) { return interleave_concat(odd_begin.high_part(), src2.high_part()); }
    float64x4 interleave_concat_high(float64x4 odd_begin, float64x4 src2) { return interleave_concat(odd_begin.high_part(), src2.high_part()); }

#if defined(_FOYE_SIMD_HAS_FP16_)
    float16x16 interleave_concat_low(float16x16 odd_begin, float16x16 src2) { return interleave_concat(odd_begin.low_part(), src2.low_part()); }
    float16x16 interleave_concat_high(float16x16 odd_begin, float16x16 src2) { return interleave_concat(odd_begin.high_part(), src2.high_part()); }
#endif
#if defined(_FOYE_SIMD_HAS_BF16_)
    bfloat16x16 interleave_concat_low(bfloat16x16 odd_begin, bfloat16x16 src2) { return interleave_concat(odd_begin.low_part(), src2.low_part()); }
    bfloat16x16 interleave_concat_high(bfloat16x16 odd_begin, bfloat16x16 src2) { return interleave_concat(odd_begin.high_part(), src2.high_part()); }
#endif
#endif
}

#endif
