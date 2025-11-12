#ifndef _FOYE_SIMDX_HPP_
#define _FOYE_SIMDX_HPP_
#pragma once

#define _FOYE_SIMD_ENABLE_EMULATED_
#define _FOYE_SIMD_ENABLE_CVTEPX64_PD_AVX2_EMULATED

// #define FOYE_SIMD_BASIC_SIMD_DEFAULT_LOAD_FROM_MEMORY_ALIGNED
#define FOYE_SIMD_ENABLE_FP16
#define FOYE_SIMD_ENABLE_BF16
#define FOYE_SIMD_DISABLE_BASIC_SIMD_SQEARE_BRACKET_ACCESS_PERFORMANCE_NOTICE
#define FOYE_SIMD_DISABLE_BASIC_MASK_SQEARE_BRACKET_ACCESS_PERFORMANCE_NOTICE
// #define FOYE_SIMD_ENABLE_SVML_TRUNC
// #define FOYE_SIMD_ENABLE_SVML_ERF
// #define FOYE_SIMD_ENABLE_SVML_HYPOT
// #define FOYE_SIMD_ENABLE_SVML_EXP2
// #define FOYE_SIMD_ENABLE_SVML_EXP
// #define FOYE_SIMD_ENABLE_SVML_CDDFNORM
// #define FOYE_SIMD_ENABLE_SVML_LOG
// #define FOYE_SIMD_ENABLE_SVML_LOG2
// #define FOYE_SIMD_ENABLE_SVML_CBRT
// #define FOYE_SIMD_ENABLE_SVML_POW
// #define FOYE_SIMD_ENABLE_SVML_LOGB
// #define FOYE_SIMD_ENABLE_SVML_SINCOS
// #define FOYE_SIMD_ENABLE_SVML_ASIN
// #define FOYE_SIMD_ENABLE_SVML_ACOS
// #define FOYE_SIMD_ENABLE_SVML_ATAN
// #define FOYE_SIMD_ENABLE_SVML_INVSQRT

#define FOYE_SIMD_ENABLE_INTERLEAVE_CONCAT_256BLH_EMULATED
#define FOYE_SIMD_ENABLE_SHIFT_EMULATED
#define FOYE_SIMD_ENABLE_COMPARISON_OPERATORS
#define FOYE_SIMD_ENABLE_EMULATED_MULTIPLIES
#define FOYE_SIMD_ENABLE_EMULATED_64BIT_MINMAX
#define FOYE_SIMD_ENABLE_EMULATED_64BIT_ABS
#define FOYE_SIMD_ENABLE_EMULATED_64BIT_SIGN_TRANSFER
#define FOYE_SIMD_ENABLE_EMULATED_AVG
#define FOYE_SIMD_ENABLE_NUMERIC_OPERATORS

#define FOYE_SIMD_ENABLE_EMULATED_MASK_STORE
// #define FOYE_SIMD_DISABLE_MASK_LOAD_ERROR

#pragma warning(push)
#pragma warning(disable : 4309)
#include "simd_def.hpp"
#include "simd_cmp.hpp"
#include "simd_opt.hpp"
#include "simd_reduce.hpp"
#include "simd_mask.hpp"
#include "simd_interleave.hpp"
#include "simd_floating.hpp"
#include "simd_cvt.hpp"
#pragma warning(pop)

#include <random>

namespace fyx::simd
{
    template<typename simd_type> requires(is_basic_simd_v<simd_type>)
    std::string format(simd_type source)
    {
        constexpr std::size_t lane_width = simd_type::lane_width;
        return [&]<std::size_t... Indices>(std::index_sequence<Indices...>)
        {
            using extract_type = typename simd_type::scalar_t;
            using format_type = std::conditional_t<is_half_basic_simd_v<simd_type>, float, extract_type>;
            std::string str{ '[' };
            ((str.append(std::format("{}{}",
                static_cast<format_type>(extract_single_from<Indices>(source)),
                (Indices == lane_width - 1) ? "]" : ", "))), ...);
            return str;
        }(std::make_index_sequence<lane_width>{});
    }

    template<typename simd_type> requires(is_basic_mask_v<simd_type>)
    std::string format(simd_type source)
    {
        constexpr std::size_t lane_width = simd_type::lane_width;
        return [&]<std::size_t... Indices>(std::index_sequence<Indices...>)
        {
            std::string str{ '[' };
            ((str.append(std::format("{}{}",
                (extract_single_from_mask<Indices>(source)
                    ? '1'
                    : '0'),
                (Indices == lane_width - 1) ? "]" : ", "))), ...);
            return str;
        }(std::make_index_sequence<lane_width>{});
    }

    template<typename simd_type>
    requires(is_basic_mask_v<simd_type> || is_basic_simd_v<simd_type>)
    void print(simd_type source)
    {
        std::cout << fyx::simd::format(source) << std::endl;
    }
}

template<typename T, std::size_t N>
void print_array(const T(&arr)[N])
{
    std::cout << "[";
    for (std::size_t i = 0; i < N; ++i)
    {
        std::cout << std::format("{}", arr[i]);
        if (i + 1 != N)
        {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
}

#endif
