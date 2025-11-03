#ifndef _FOYE_BFLOAT16_HPP_
#define _FOYE_BFLOAT16_HPP_
#pragma once

#include <type_traits>
#include <bit>
#include <cstdint>
#include <cmath>
#include <utility>
#include <limits>

namespace fy
{
    struct bfloat16
    {
        bfloat16() noexcept = default;

        bfloat16(float from_f32) noexcept
        {
            std::uint32_t u32_val = std::bit_cast<std::uint32_t>(from_f32);
            if (((u32_val & exponent_mask) == exponent_mask) && (u32_val & mantissa_mask))
            {
                bits_ = static_cast<std::uint16_t>(u32_val >> 16);
                if ((bits_ & 0x007F) == 0)
                {
                    bits_ |= 0x0001;
                }
            }
            else
            {
                bits_ = static_cast<std::uint16_t>(u32_val >> 16);
            }
        }

        operator float() const noexcept
        {
            return std::bit_cast<float>(static_cast<std::uint32_t>(bits_) << 16);
        }

        bfloat16 operator-() const noexcept
        {
            bfloat16 result;
            result.bits_ = bits_ ^ 0x8000;
            return result;
        }

        static bfloat16 bfloatFromBits(std::uint16_t w)
        {
            std::uint32_t u32_val = static_cast<std::uint32_t>(w) << 16;

            if (((u32_val & exponent_mask) == exponent_mask) && (u32_val & mantissa_mask))
            {
                u32_val |= 0x00400000;
                u32_val &= ~0x00200000;
            }
            else if ((u32_val & exponent_mask) == exponent_mask)
            {
                u32_val &= ~mantissa_mask;
            }

            bfloat16 result;
            result.bits_ = static_cast<std::uint16_t>(u32_val >> 16);
            return result;
        }

        bfloat16 operator + (bfloat16 other) const { return round(static_cast<float>(*this) + static_cast<float>(other)); }
        bfloat16 operator - (bfloat16 other) const { return round(static_cast<float>(*this) - static_cast<float>(other)); }
        bfloat16 operator * (bfloat16 other) const { return round(static_cast<float>(*this) * static_cast<float>(other)); }
        bfloat16 operator / (bfloat16 other) const { return round(static_cast<float>(*this) / static_cast<float>(other)); }

        bfloat16& operator += (bfloat16 other) { *this = *this + other; return *this; }
        bfloat16& operator -= (bfloat16 other) { *this = *this - other; return *this; }
        bfloat16& operator *= (bfloat16 other) { *this = *this * other; return *this; }
        bfloat16& operator /= (bfloat16 other) { *this = *this / other; return *this; }

        bool operator == (bfloat16 other) const { return bits_ == other.bits_; }
        bool operator != (bfloat16 other) const { return bits_ != other.bits_; }

        bool operator < (bfloat16 other) const { return static_cast<float>(*this) < static_cast<float>(other); }
        bool operator > (bfloat16 other) const { return static_cast<float>(*this) > static_cast<float>(other); }
        bool operator <= (bfloat16 other) const { return static_cast<float>(*this) <= static_cast<float>(other); }
        bool operator >= (bfloat16 other) const { return static_cast<float>(*this) >= static_cast<float>(other); }

        static bfloat16 abs(bfloat16 value)
        {
            return bfloat16::bfloatFromBits(value.bits_ & 0x7FFF);
        }

        static bool isnan(bfloat16 value)
        {
            return ((value.bits_ & bfloat16::exponent_mask) == bfloat16::exponent_mask) &&
                ((value.bits_ & bfloat16::mantissa_mask) != 0);
        }

        static bfloat16 round(double x)
        {
            bfloat16 maxval = bfloat16::bfloatFromBits(0x0080);
            bfloat16 lowestval = bfloat16::bfloatFromBits(0xFF7F);

            if (x != x) { return bfloat16::bfloatFromBits(0x7FC0); }
            if (x > static_cast<double>(maxval)) return maxval;
            if (x < static_cast<double>(lowestval)) return lowestval;
            if (x == std::numeric_limits<double>::infinity()) return bfloat16::bfloatFromBits(0x7F80);
            if (x == -std::numeric_limits<double>::infinity()) return bfloat16::bfloatFromBits(0xFF80);

            double int_part;
            double frac_part = std::modf(x, &int_part);

            if (x >= 0)
            {
                if (frac_part > 0.5)
                {
                    int_part += 1.0;
                }
                else if (frac_part == 0.5)
                {
                    if (std::fmod(int_part, 2.0) != 0.0)
                    {
                        int_part += 1.0;
                    }
                }
            }
            else
            {
                if (frac_part < -0.5)
                {
                    int_part -= 1.0;
                }
                else if (frac_part == -0.5)
                {
                    if (std::fmod(int_part, 2.0) != 0.0)
                    {
                        int_part -= 1.0;
                    }
                }
            }

            return bfloat16(static_cast<float>(int_part));
        }

        static constexpr std::uint32_t exponent_mask = 0x7F800000;
        static constexpr std::uint32_t mantissa_mask = 0x007FFFFF;
        static constexpr std::uint16_t non_sign_mask = 0x7FFF;
        std::uint16_t bits_;
    };
}

#endif