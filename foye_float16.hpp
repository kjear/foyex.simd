#ifndef _FOYE_FLOAT16_HPP_
#define _FOYE_FLOAT16_HPP_
#pragma once

#include <type_traits>
#include <bit>
#include <cstdint>
#include <cmath>
#include <utility>

#define _FOYE_ENABLE_INTRIN_F16C_

namespace fy
{
    struct float16
    {
        float16() noexcept = default;
        
        float16(float x) noexcept 
#if defined(_FOYE_ENABLE_INTRIN_F16C_)
            : bits_(static_cast<std::uint16_t>(_mm_extract_epi16(
                _mm_cvtps_ph(_mm_set_ss(x), _MM_FROUND_TO_NEAREST_INT), 0))) { }
#else
        {
            std::uint32_t in = std::bit_cast<std::uint32_t>(x);
            std::uint32_t sign = in & F32_SIGN_MASK;
            in ^= sign;

            if (in >= F32_OVERFLOW_THRESHOLD)
            {
                bits_ = (in > F32_NAN_VALUE)
                    ? static_cast<std::uint16_t>(F16_NAN_MASK)
                    : static_cast<std::uint16_t>(F16_MAX_EXPONENT);
            }
            else
            {
                if (in < F32_DENORMAL_THRESHOLD)
                {
                    float temp = std::bit_cast<float>(in) + 0.5f;
                    in = std::bit_cast<std::uint32_t>(temp);
                    bits_ = static_cast<std::uint16_t>(in - F32_HALF_CONST);
                }
                else
                {
                    std::uint32_t t = in + F32_ROUND_CONST;
                    bits_ = static_cast<std::uint16_t>(
                        (t + ((in >> F32_EXP_SHIFT) & 1)) >> F32_EXP_SHIFT);
                }
            }

            bits_ = static_cast<std::uint16_t>(bits_ | (sign >> F32_SIGN_SHIFT));
        }
#endif

        operator float() const noexcept
        {
#if defined(_FOYE_ENABLE_INTRIN_F16C_)
            return _mm_cvtss_f32(_mm_cvtph_ps(
                _mm_set_epi16(0, 0, 0, 0, 0, 0, 0, 
                    std::bit_cast<short>(bits_)
                )));
#else
            std::uint32_t t = ((bits_ & F16_NON_SIGN_MASK) << F32_EXP_SHIFT) + F32_MAGIC_38000000;
            std::uint32_t sign = (bits_ & F16_SIGN_MASK) << F32_SIGN_SHIFT;
            std::uint32_t e = bits_ & F16_EXPONENT_MASK;

            std::uint32_t result = t + F32_ONE_SHIFT_23;
            if (e >= F16_MAX_EXPONENT)
            {
                result = t + F32_MAGIC_38000000;
            }
            else if (e == F16_ZERO_EXPONENT)
            {
                float temp = std::bit_cast<float>(result);
                temp -= F16_DENORMAL_FACTOR;
                result = std::bit_cast<std::uint32_t>(temp);
            }
            else
            {
                result = t;
            }
            result |= sign;
            return std::bit_cast<float>(result);
#endif
        }

        float16 operator-() const noexcept
        {
            if (isnan(*this))
            {
                return *this;
            }

            if ((bits_ & F16_NON_SIGN_MASK) == 0)
            {
                float16 result;
                result.bits_ = F16_SIGN_MASK;
                return result;
            }

            float16 result;
            result.bits_ = bits_ ^ F16_SIGN_MASK;
            return result;
        }

        static float16 hfloatFromBits(std::uint16_t w) noexcept
        {
            std::uint32_t t = ((w & F16_NON_SIGN_MASK) << F32_EXP_SHIFT) + F32_MAGIC_38000000;
            std::uint32_t sign = (w & F16_SIGN_MASK) << F32_SIGN_SHIFT;
            std::uint32_t e = w & F16_EXPONENT_MASK;

            std::uint32_t result_bits = t + F32_ONE_SHIFT_23;
            if (e >= F16_MAX_EXPONENT)
            {
                result_bits = t + F32_MAGIC_38000000;
            }
            else if (e == F16_ZERO_EXPONENT)
            {
                std::uint32_t temp = t + F32_ONE_SHIFT_23;
                float float_temp = std::bit_cast<float>(temp);
                float_temp -= F16_DENORMAL_FACTOR;
                result_bits = std::bit_cast<std::uint32_t>(float_temp);
            }
            else
            {
                result_bits = t;
            }

            result_bits |= sign;
            return float16(std::bit_cast<float>(result_bits));
        }

        static float16 abs(float16 value)
        {
            float16 result;
            result.bits_ = value.bits_ & F16_NON_SIGN_MASK;
            return result;
        }

        static bool isnan(float16 value)
        {
            return (value.bits_ & F16_EXPONENT_MASK) == F16_MAX_EXPONENT &&
                (value.bits_ & F16_MANTISSA_MASK) != 0;
        }

        static float16 round(float value)
        {
            std::uint32_t u = std::bit_cast<std::uint32_t>(value);
            std::uint32_t sign = u & F32_SIGN_MASK;
            u ^= sign;

            float16 result{};

            if (u >= F32_OVERFLOW_THRESHOLD)
            {
                result.bits_ = (u > F32_NAN_VALUE)
                    ? F16_NAN_MASK
                    : F16_MAX_EXPONENT;
            }
            else
            {
                if (u < F32_DENORMAL_THRESHOLD)
                {
                    float rounded = std::round(value * 0x1p+24f) * 0x1p-24f;
                    return float16(rounded);
                }
                else
                {
                    std::uint32_t t = u + F32_ROUND_CONST;
                    result.bits_ = static_cast<std::uint16_t>((t + ((u >> F32_EXP_SHIFT) & 1)) >> F32_EXP_SHIFT);
                }
            }

            result.bits_ |= static_cast<std::uint16_t>(sign >> F32_SIGN_SHIFT);
            return result;
        }

        std::uint16_t bits_;

        static constexpr std::uint32_t F32_SIGN_MASK = 0x80000000;
        static constexpr std::uint32_t F32_EXPONENT_MASK = 0x7F800000;
        static constexpr std::uint32_t F32_MANTISSA_MASK = 0x007FFFFF;
        static constexpr std::uint32_t F32_OVERFLOW_THRESHOLD = 0x47800000;
        static constexpr std::uint32_t F32_DENORMAL_THRESHOLD = 0x38800000;
        static constexpr std::uint32_t F32_ROUND_CONST = 0xC8000FFF;
        static constexpr std::uint32_t F32_HALF_CONST = 0x3F000000;
        static constexpr std::uint32_t F32_NAN_VALUE = 0x7F800000;
        static constexpr std::uint32_t F32_EXP_SHIFT = 13;
        static constexpr std::uint32_t F32_SIGN_SHIFT = 16;
        static constexpr std::uint32_t F32_MAGIC_38000000 = 0x38000000;
        static constexpr std::uint32_t F32_ONE_SHIFT_23 = 1 << 23;

        static constexpr std::uint16_t F16_SIGN_MASK = 0x8000;
        static constexpr std::uint16_t F16_NON_SIGN_MASK = 0x7FFF;
        static constexpr std::uint16_t F16_EXPONENT_MASK = 0x7C00;
        static constexpr std::uint16_t F16_MANTISSA_MASK = 0x03FF;
        static constexpr std::uint16_t F16_INFINITY_EXP = 0x7C00;
        static constexpr std::uint16_t F16_NAN_MASK = 0x7E00;
        static constexpr std::uint16_t F16_MAX_EXPONENT = 0x7C00;
        static constexpr std::uint16_t F16_ZERO_EXPONENT = 0x0000;
        static constexpr float F16_DENORMAL_FACTOR = 6.103515625e-05f;

    };

    float16 operator + (const float16& lhs, const float16& rhs) noexcept
    {
        return float16::round(lhs.operator float() + rhs.operator float());
    }

    float16 operator - (const float16& lhs, const float16& rhs) noexcept
    {
        return float16::round(lhs.operator float() - rhs.operator float());
    }

    float16 operator * (const float16& lhs, const float16& rhs) noexcept
    {
        return float16::round(lhs.operator float() * rhs.operator float());
    }

    float16 operator / (const float16& lhs, const float16& rhs) noexcept
    {
        return float16::round(lhs.operator float() / rhs.operator float());
    }

    float16& operator += (float16& lhs, const float16& rhs) noexcept
    {
        lhs = lhs + rhs; return lhs;
    }

    float16& operator -= (float16& lhs, const float16& rhs) noexcept
    {
        lhs = lhs - rhs; return lhs;
    }

    float16& operator *= (float16& lhs, const float16& rhs) noexcept
    {
        lhs = lhs * rhs; return lhs;
    }

    float16& operator /= (float16& lhs, const float16& rhs) noexcept
    {
        lhs = lhs / rhs; return lhs;
    }

    bool operator == (const float16& lhs, const float16& rhs) noexcept
    {
        return lhs.bits_ == rhs.bits_;
    }

    bool operator != (const float16& lhs, const float16& rhs) noexcept
    {
        return lhs.bits_ != rhs.bits_;
    }

    bool operator < (const float16& lhs, const float16& rhs) noexcept
    {
        return (lhs.operator float()) < (rhs.operator float());
    }

    bool operator > (const float16& lhs, const float16& rhs) noexcept
    {
        return (lhs.operator float()) > (rhs.operator float());
    }

    bool operator <= (const float16& lhs, const float16& rhs) noexcept
    {
        return (lhs.operator float()) <= (rhs.operator float());
    }

    bool operator >= (const float16& lhs, const float16& rhs) noexcept
    {
        return (lhs.operator float()) >= (rhs.operator float());
    }
}
#endif