#ifndef _FOYE_SIMDX_HPP_
#define _FOYE_SIMDX_HPP_
#pragma once

#define FOYE_SIMD_DISABLE_PERFORMANCE_NOTICE
#define _FOYE_SIMD_ENABLE_EMULATED_
#define _FOYE_SIMD_ENABLE_CVTEPX64_PD_AVX2_EMULATED
//#define FOYE_SIMD_ENABLE_SVML
#include "simd_def.hpp"
#include "simd_cmp.hpp"
#include "simd_opt.hpp"
#include "simd_reduce.hpp"
#include "simd_mask.hpp"
#include "simd_interleave.hpp"
#include "simd_floating.hpp"
#include "simd_cvt.hpp"



#include <random>

namespace fyx::simd::test
{
	template<typename vector_type> requires(std::is_integral_v<typename vector_type::scalar_t>)
	vector_type create_random_vector()
	{
		int lowest, max;
		if constexpr (std::_Is_any_of_v<typename vector_type::scalar_t, 
			std::uint32_t, std::int64_t, std::uint64_t>)
		{
			if constexpr (std::is_same_v<typename vector_type::scalar_t, std::int64_t>)
			{
				lowest = std::numeric_limits<int>::lowest();
			}
			else
			{
				lowest = 0;
			}
			max = std::numeric_limits<int>::max();
		}
		else
		{
			lowest = std::numeric_limits<typename vector_type::scalar_t>::lowest();
			max = std::numeric_limits<typename vector_type::scalar_t>::max();
		}

		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_int_distribution<int> dist(lowest, max);

		typename vector_type::scalar_t arr[vector_type::lane_width] = { };

		for (std::size_t i = 0; i < vector_type::lane_width; ++i)
		{
			arr[i] = dist(gen);
		}

		return fyx::simd::load_unaligned<vector_type>(arr);
	}

	template<typename vector_type> 
		requires(std::is_floating_point_v<typename vector_type::scalar_t>
#ifdef _FOYE_SIMD_HAS_FP16_
	|| std::is_same_v<typename vector_type::scalar_t, fy::float16>
#endif
#ifdef _FOYE_SIMD_HAS_BF16_
		|| std::is_same_v<typename vector_type::scalar_t, fy::bfloat16>
#endif
			)
	vector_type create_random_vector()
	{
		using scalar_t = typename vector_type::scalar_t;
		using use_type = std::conditional_t<std::is_same_v<scalar_t, double>, double, float>;

		std::random_device rd;
		std::mt19937 gen(rd());

		scalar_t arr[vector_type::lane_width] = { };
		std::uniform_real_distribution<use_type> dist(-10.0, 10.0);
		for (std::size_t i = 0; i < vector_type::lane_width; ++i)
		{
			arr[i] = static_cast<scalar_t>(dist(gen));
		}

		return fyx::simd::load_unaligned<vector_type>(arr);
	}

}


namespace fyx::simd
{
    template<std::size_t count_element, std::size_t bits_width>
    void print(basic_mask<count_element, bits_width> vec)
    {
        constexpr std::size_t single_width_bits = bits_width / count_element;
        constexpr std::size_t single_width = single_width_bits / CHAR_BIT;

        using fmt_t = std::conditional_t<
            single_width == 1,
            std::uint8_t,
            std::conditional_t<
            single_width == 2,
            std::uint16_t,
            std::conditional_t<
            single_width == 4, std::uint32_t, std::uint64_t>>
            >;

        alignas(alignof(typename basic_mask<count_element, bits_width>::vector_t))
            fmt_t temp[count_element] = {};

        fyx::simd::detail::store_aligned<typename basic_mask<count_element, bits_width>::vector_t>(vec.data, temp);
        std::cout << "[";
        for (std::size_t i = 0; i < count_element; ++i)
        {
            auto f = std::format("{}",
                (temp[i] != fmt_t{ 0 })
                ? "1"
                : "0"
            );
            std::cout << f;

            if (i + 1 != count_element)
            {
                std::cout << ", ";
            }
        }
        std::cout << "]" << std::endl;
    }

    template<typename T, std::size_t bits_width>
    void print(basic_simd<T, bits_width> vec)
    {
        constexpr std::size_t lane_width = basic_simd<T, bits_width>::lane_width;
        std::cout << "[";
        for (std::size_t i = 0; i < lane_width; ++i)
        {
            auto print_value = [](auto value)
                {
                    using ValueType = decltype(value);
                    if constexpr (std::is_same_v<ValueType, float> ||
                        std::is_same_v<ValueType, double> ||
                        std::is_integral_v<ValueType>)
                    {
                        return std::format("{}", value);
                    }
#ifdef _FOYE_SIMD_HAS_FP16_
                    else if constexpr (std::is_same_v<ValueType, fy::float16>)
                    {
                        return std::format("{}", static_cast<float>(value));
                    }
#endif
#ifdef _FOYE_SIMD_HAS_BF16_
                    else if constexpr (std::is_same_v<ValueType, fy::bfloat16>)
                    {
                        return std::format("{}", static_cast<float>(value));
                    }
#endif
                    else
                    {
                        return std::format("{}", value);
                    }
                };

            std::cout << print_value(vec[i]);

            if (i + 1 != lane_width)
            {
                std::cout << ", ";
            }
        }
        std::cout << "]" << std::endl;
    }
}





#endif