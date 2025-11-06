#ifndef _FOYE_SIMD_DEF_HPP_
#define _FOYE_SIMD_DEF_HPP_
#pragma once

#include "simd_utility.hpp"

namespace fyx::simd
{
    bool is_AVX512F_available()
    {
        std::int32_t reg[4];
        __cpuidex(reg, 7, 0);
        return (reg[1] & (1 << 16)) != 0;
    }

    bool is_AVX2_available()
    {
        std::int32_t reg[4];
        __cpuidex(reg, 7, 0);
        return (reg[1] & (1 << 5)) != 0;
    }

    bool is_FMA_available()
    {
        std::int32_t reg[4];
        __cpuidex(reg, 1, 0);
        return (reg[2] & (1 << 12)) != 0;
    }

    bool is_SSE2_available()
    {
        std::int32_t reg[4];
        __cpuidex(reg, 1, 0);
        return (reg[3] & (1 << 26)) != 0;
    }

    bool is_SSE3_available()
    {
        std::int32_t reg[4];
        __cpuidex(reg, 1, 0);
        return (reg[2] & (1 << 0)) != 0;
    }

    bool is_SSE4_1_available()
    {
        std::int32_t reg[4];
        __cpuidex(reg, 1, 0);
        return (reg[2] & (1 << 19)) != 0;
    }

    int is_AVX_available()
    {
        std::int32_t reg[4];
        __cpuidex(reg, 1, 0);
        return (reg[2] & (1 << 28)) != 0;
    }

    void flush_cache_line(const void* addr) { _mm_clflush(addr); }
    void load_fence() { _mm_lfence(); }
    void store_fence() { _mm_sfence(); }
    void memory_fence() { _mm_mfence(); }

    void zero_upper() { _mm256_zeroupper(); }
    void zero_all() { _mm256_zeroall(); }

    struct all_zero_guard { ~all_zero_guard() { _mm256_zeroall(); } };
    struct upper_zero_guard { ~upper_zero_guard() { _mm256_zeroupper(); } };

    template<typename T>
    void nontemporal_hint(T* ptr)
    {
        _mm_prefetch(reinterpret_cast<const char*>(ptr), _MM_HINT_NTA);
    }

    enum class prefetch_strategy
    {
        prefetch_nta,
        prefetch_t0,
        prefetch_t1,
        prefetch_t2
    };

    template<prefetch_strategy Strategy = prefetch_strategy::prefetch_t0>
    void prefetch(const void* addr)
    {
        switch (Strategy)
        {
        case prefetch_strategy::prefetch_nta:
            _mm_prefetch(static_cast<const char*>(addr), _MM_HINT_NTA);
            break;
        case prefetch_strategy::prefetch_t0:
            _mm_prefetch(static_cast<const char*>(addr), _MM_HINT_T0);
            break;
        case prefetch_strategy::prefetch_t1:
            _mm_prefetch(static_cast<const char*>(addr), _MM_HINT_T1);
            break;
        case prefetch_strategy::prefetch_t2:
            _mm_prefetch(static_cast<const char*>(addr), _MM_HINT_T2);
            break;
        }
    }

    namespace internel
    {
        struct CPUInfo 
        {
            char brand_string[48];
            char newline;
            char null_term;
            char called_flag;
        };

        static CPUInfo cpu_info = { 0 };
    }

    const char* cpuid_string()
    {
        if (internel::cpu_info.called_flag == 1) 
        {
            return internel::cpu_info.brand_string;
        }

        union
        {
            std::int32_t info[4];
            std::uint8_t str[16];
        } u;
        int i, j;

        for (i = 0; i < 3; i++)
        {
            __cpuidex(u.info, i + 0x80000002, 0);

            for (j = 0; j < 16; j++)
            {
                internel::cpu_info.brand_string[i * 16 + j] = u.str[j];
            }
        }

        internel::cpu_info.newline = '\n';
        internel::cpu_info.null_term = '\0';
        internel::cpu_info.called_flag = 1;

        return std::as_const(internel::cpu_info.brand_string);
    }


    template<std::size_t count_element, std::size_t bits_width>
    struct alignas(bits_width / CHAR_BIT) basic_mask;

    template<typename T, std::size_t bits_width>
    struct alignas(bits_width / CHAR_BIT) basic_simd;


    template<typename T, std::size_t bits_width>
    struct alignas(bits_width / CHAR_BIT) basic_simd
    {
        static_assert(bits_width == 128 || bits_width == 256);

        static_assert(bits_width % (sizeof(T) * CHAR_BIT) == 0,
            "Bit width must be divisible by element size");

        static constexpr std::size_t bit_width = bits_width;
        static constexpr std::size_t lane_width = bits_width / CHAR_BIT / sizeof(T);
        
        static constexpr std::size_t scalar_bit_width = sizeof(T) * CHAR_BIT;

        using scalar_t = T;
        using vector_t = std::conditional_t<bits_width == 128,
            fyx::simd::detail::vector_128_t<scalar_t>
            , fyx::simd::detail::vector_256_t<scalar_t>>;

        vector_t data;

        basic_simd() noexcept = default;
        explicit basic_simd(vector_t data_) noexcept : data(data_) { }

        explicit basic_simd(basic_simd<T, bits_width / 2> low, basic_simd<T, bits_width / 2> high) noexcept
            : data(fyx::simd::detail::merge(low.data, high.data)) { }

        explicit basic_simd(scalar_t brocast_) noexcept
            : data(fyx::simd::detail::brocast<vector_t, scalar_t>(brocast_)) { }

        explicit basic_simd(const scalar_t* mem_addr) noexcept
#if defined(_FOYE_SIMD_DEFAULT_ALIGNED_)
            : data(fyx::simd::detail::load_aligned<vector_t>(mem_addr)) { }
#else
            : data(fyx::simd::detail::load_unaligned<vector_t>(mem_addr)) { }
#endif

        operator vector_t() const noexcept { return data; }

        basic_simd<T, bits_width / 2> low_part() const noexcept
        { return basic_simd<T, bits_width / 2>{fyx::simd::detail::split_low(this->data)}; }

        basic_simd<T, bits_width / 2> high_part() const noexcept
        { return basic_simd<T, bits_width / 2>{fyx::simd::detail::split_high(this->data)}; }

        FOYE_SIMD_PERFORMANCE_MATTER scalar_t operator [] (std::size_t index) const noexcept
        {
            scalar_t temp[lane_width] = { };
            fyx::simd::detail::store_unaligned(this->data, temp);
            return temp[index];
        }

        basic_mask<lane_width, bits_width> as_basic_mask() const noexcept
        {
            return basic_mask<lane_width, bits_width>{ *this };
        }
    };

    template<std::size_t count_element, std::size_t bits_width>
    struct alignas(bits_width / CHAR_BIT) basic_mask
    {
        static_assert(bits_width == 128 || bits_width == 256);

        static constexpr std::size_t bit_width = bits_width;
        static constexpr std::size_t lane_width = count_element;
        static constexpr std::size_t single_width_bits = bits_width / count_element;
        static constexpr std::size_t single_width_bytes = single_width_bits / CHAR_BIT;

        using vector_t = std::conditional_t<bits_width == 128, __m128i, __m256i>;

        basic_mask() noexcept = default;

        template<typename T>
        basic_mask(basic_simd<T, bits_width> from_simd) noexcept 
            : data(fyx::simd::detail::basic_reinterpret<vector_t, 
                typename basic_simd<T, bits_width>::vector_t>(from_simd.data)) { }

        template<typename input_type> 
        requires(sizeof(input_type) == sizeof(vector_t) && fyx::simd::detail::is_mm_vector_type_v<input_type>)
        basic_mask(input_type input) noexcept
            : data(fyx::simd::detail::basic_reinterpret<vector_t, input_type>(input)) { }

        template<typename output_type> 
        requires(sizeof(output_type) == sizeof(vector_t) && fyx::simd::detail::is_mm_vector_type_v<output_type>)
        operator output_type() const noexcept
        {
            return fyx::simd::detail::basic_reinterpret<output_type, vector_t>(this->data);
        }

        template<typename simd_type>
        requires(simd_type::lane_width == lane_width && simd_type::bit_width == bit_width)
        simd_type as_basic_simd() const noexcept
        {
            return simd_type{ fyx::simd::detail::basic_reinterpret<
                typename simd_type::vector_t>(this->data) };
        }

        FOYE_SIMD_PERFORMANCE_MATTER bool operator [] (std::size_t index) const noexcept
        {
            using fmt_t = std::conditional_t<
                single_width_bytes == 1,
                std::uint8_t,
                std::conditional_t<
                single_width_bytes == 2,
                std::uint16_t,
                std::conditional_t<
                single_width_bytes == 4, std::uint32_t, std::uint64_t>>
                >;

            alignas(alignof(vector_t)) fmt_t temp[count_element] = {};
            fyx::simd::detail::store_unaligned(this->data, temp);
            return (temp[index] != fmt_t{ 0 });
        }

        vector_t data;
    };

    namespace detail
    {
        template<typename>
        struct is_basic_simd : std::false_type {};

        template<typename T, std::size_t bits_width>
        struct is_basic_simd<basic_simd<T, bits_width>> : std::true_type {};
    }

    template<typename T>
    constexpr bool is_basic_simd_v = detail::is_basic_simd<T>::value;

    template<typename T>
    constexpr bool is_128bits_mask_v = (
        std::is_same_v<T, basic_mask<16, 128>> || std::is_same_v<T, basic_mask<8, 128>> || 
        std::is_same_v<T, basic_mask<4, 128>> || std::is_same_v<T, basic_mask<2, 128>>);

    template<typename T>
    constexpr bool is_256bits_mask_v = (
        std::is_same_v<T, basic_mask<32, 256>> || std::is_same_v<T, basic_mask<16, 256>> ||
        std::is_same_v<T, basic_mask<8, 256>> || std::is_same_v<T, basic_mask<4, 256>>);

    template<typename T>
    constexpr bool is_basic_mask_v = (fyx::simd::is_128bits_mask_v<T> || fyx::simd::is_256bits_mask_v<T>);

    template<typename T> requires(fyx::simd::is_basic_simd_v<T>)
    constexpr bool is_256bits_simd_v = (sizeof(typename T::vector_t) == sizeof(__m256i));

    template<typename T> requires(fyx::simd::is_basic_simd_v<T>)
    constexpr bool is_128bits_simd_v = (sizeof(typename T::vector_t) == sizeof(__m128i));

    template<typename T> requires(fyx::simd::is_basic_simd_v<T>)
    constexpr bool is_integral_basic_simd_v = std::is_integral_v<typename T::scalar_t>;

    template<typename T> requires(fyx::simd::is_basic_simd_v<T>)
    constexpr bool is_unsigned_integral_basic_simd_v = (fyx::simd::is_integral_basic_simd_v<T>
            && std::is_unsigned_v<typename T::scalar_t>);

    template<typename T> requires(fyx::simd::is_basic_simd_v<T>)
    constexpr bool is_signed_integral_basic_simd_v = (fyx::simd::is_integral_basic_simd_v<T>
            && std::is_signed_v<typename T::scalar_t>);

    template<typename T> requires(fyx::simd::is_basic_simd_v<T>)
    constexpr bool is_floating_basic_simd_v = (std::is_floating_point_v<typename T::scalar_t>
#if defined(_FOYE_SIMD_HAS_FP16_)
        || std::is_same_v<typename T::scalar_t, fy::float16>
#endif
#if defined(_FOYE_SIMD_HAS_BF16_)
        || std::is_same_v<typename T::scalar_t, fy::bfloat16>
#endif
        );

#if defined(_FOYE_SIMD_HAS_FP16_) || defined(_FOYE_SIMD_HAS_BF16_)
    template<typename T> requires(fyx::simd::is_basic_simd_v<T>)
    constexpr bool is_half_basic_simd_v =
#if defined(_FOYE_SIMD_HAS_FP16_) && defined(_FOYE_SIMD_HAS_BF16_)
        (std::is_same_v<typename T::scalar_t, fy::float16> || std::is_same_v<typename T::scalar_t, fy::bfloat16>);
#elif defined(_FOYE_SIMD_HAS_FP16_)
        std::is_same_v<typename T::scalar_t, fy::float16>;
#elif defined(_FOYE_SIMD_HAS_BF16_)
        std::is_same_v<typename T::scalar_t, fy::bfloat16>;
#endif
#else
    template<typename T> requires(fyx::simd::is_basic_simd_v<T>)
    constexpr bool is_half_basic_simd_v = false;
#endif

    template<typename simd_type> requires(fyx::simd::is_integral_basic_simd_v<simd_type>)
    using as_unsigned_type = std::conditional_t<
        std::is_unsigned_v<typename simd_type::scalar_t>,
        simd_type,
        fyx::simd::basic_simd<
            std::make_unsigned_t<typename simd_type::scalar_t>,
            simd_type::bit_width
        >
    >;

    template<typename simd_type> requires(fyx::simd::is_integral_basic_simd_v<simd_type>)
    using as_signed_type = std::conditional_t<
        std::is_signed_v<typename simd_type::scalar_t>,
        simd_type,
        fyx::simd::basic_simd<
            std::make_signed_t<typename simd_type::scalar_t>,
            simd_type::bit_width
        >
    >;

    template<typename simd_type> requires(fyx::simd::is_basic_simd_v<simd_type>)
    simd_type allzero_bits_as()
    {
        return simd_type{ fyx::simd::detail::zero_vec<typename simd_type::vector_t>() };
    }

    template<typename simd_type> requires(fyx::simd::is_basic_simd_v<simd_type>)
    simd_type allone_bits_as()
    {
        return simd_type{ fyx::simd::detail::one_vec<typename simd_type::vector_t>() };
    }

    template<typename simd_type, typename brocast_scalar_type> 
    requires(fyx::simd::is_basic_simd_v<simd_type> && std::is_convertible_v<typename simd_type::scalar_t, brocast_scalar_type>)
    simd_type load_brocast(brocast_scalar_type value)
    {
        return simd_type{ static_cast<typename simd_type::scalar_t>(value) };
    }

    template<typename simd_type> requires(fyx::simd::is_basic_simd_v<simd_type>)
    simd_type load_aligned(const typename simd_type::scalar_t* mem_addr)
    {
        return simd_type{ fyx::simd::detail::load_aligned<typename simd_type::vector_t>(mem_addr) };
    }

    template<typename simd_type> requires(fyx::simd::is_basic_simd_v<simd_type>)
    simd_type load_unaligned(const typename simd_type::scalar_t* mem_addr)
    {
        return simd_type{ fyx::simd::detail::load_unaligned<typename simd_type::vector_t>(mem_addr) };
    }

    template<typename simd_type> requires(fyx::simd::is_basic_simd_v<simd_type>)
    void store_aligned(simd_type to_store_vector, void* mem_addr)
    {
        fyx::simd::detail::store_aligned<typename simd_type::vector_t>(
            to_store_vector.data,
            reinterpret_cast<typename simd_type::scalar_t*>(mem_addr)
        );
    }

    template<typename simd_type> requires(fyx::simd::is_basic_simd_v<simd_type>)
    void store_unaligned(simd_type to_store_vector, void* mem_addr)
    {
        fyx::simd::detail::store_unaligned<typename simd_type::vector_t>(
            to_store_vector.data,
            reinterpret_cast<typename simd_type::scalar_t*>(mem_addr)
        );
    }

    template<typename simd_type> requires(fyx::simd::is_basic_simd_v<simd_type>)
    void store_stream(simd_type to_store_vector, void* mem_addr)
    {
        fyx::simd::detail::store_stream<typename simd_type::vector_t>(
            to_store_vector.data,
            reinterpret_cast<typename simd_type::scalar_t*>(mem_addr)
        );
    }
    
    template<typename simd_type, typename ... Args> 
    requires(fyx::simd::is_basic_simd_v<simd_type> && sizeof...(Args) <= simd_type::lane_width
        && (std::is_constructible_v<typename simd_type::scalar_t, Args> && ...))
    FOYE_SIMD_PERFORMANCE_MATTER simd_type load_by_each(Args&& ... args)
    {
        using input_scalar_t = typename simd_type::scalar_t;
        using input_vector_t = typename simd_type::vector_t;
        
        using setter_invoker = fyx::simd::detail::setter_by_each_invoker<input_vector_t, sizeof(input_scalar_t),
            simd_type::lane_width>;

        alignas(alignof(input_vector_t)) input_scalar_t temp[simd_type::lane_width];
        auto args_tuple = std::forward_as_tuple(std::forward<Args>(args)...);

        [&]<std::size_t... index>(std::index_sequence<index...>)
        {
            ((temp[index] = [&]()
            {
                if constexpr (index < sizeof...(Args))
                {
#if defined(_FOYE_SIMD_HAS_FP16_)
                    if constexpr (std::is_same_v<input_scalar_t, fy::float16>)
                    {
                        return fy::float16(static_cast<float>(std::get<index>(args_tuple)));
                    }
#endif
#if defined(_FOYE_SIMD_HAS_BF16_)
                    if constexpr (std::is_same_v<input_scalar_t, fy::bfloat16>)
                    {
                        return fy::bfloat16(static_cast<float>(std::get<index>(args_tuple)));
                    }
#endif
                    return static_cast<input_scalar_t>(std::get<index>(args_tuple));
                }
                else
                {
                    return input_scalar_t{ 0 };
                }
            }()), ...);
        }(std::make_index_sequence<simd_type::lane_width>{});

        setter_invoker invoker{};

        return simd_type{
            [&] <std::size_t... Is>(std::index_sequence<Is...>)
            {
#if defined(_FOYE_SIMD_HAS_FP16_)
            if constexpr (std::is_same_v<input_scalar_t, fy::float16>)
            {
                return invoker(std::bit_cast<std::uint16_t>(temp[Is])...);
            }
#endif
#if defined(_FOYE_SIMD_HAS_BF16_)
            if constexpr (std::is_same_v<input_scalar_t, fy::bfloat16>)
            {
                return invoker(std::bit_cast<std::uint16_t>(temp[Is])...);
            }
#endif
            return invoker(temp[Is]...);
            }(std::make_index_sequence<simd_type::lane_width>{})
        };
    }

    template<typename target_simd, typename source_scalar, std::size_t bits_width> 
    requires(fyx::simd::is_basic_simd_v<target_simd> && (target_simd::bit_width == bits_width))
    target_simd reinterpret(basic_simd<source_scalar, bits_width> source_vec)
    {
        return target_simd{ fyx::simd::detail::basic_reinterpret<typename target_simd::vector_t,
            typename fyx::simd::basic_simd<source_scalar, bits_width>::vector_t>(source_vec.data) };
    }

    template<typename mask_type, typename ... Args> 
    requires(fyx::simd::is_basic_mask_v<mask_type> && (std::is_constructible_v<Args, bool> && ...)
        && (sizeof...(Args) <= mask_type::lane_width))
    FOYE_SIMD_PERFORMANCE_MATTER mask_type load_mask_by_each(Args&& ... args)
    {
        constexpr std::size_t single_width_bits = mask_type::bit_width / mask_type::lane_width;
        constexpr std::size_t single_width = single_width_bits / CHAR_BIT;

        using bits_contain_type = std::conditional_t<
            single_width == 1, std::uint8_t,
                std::conditional_t<
                single_width == 2, std::uint16_t,
                    std::conditional_t<
                    single_width == 4, std::uint32_t, std::uint64_t>
                >
            >;

        using setter_invoker = fyx::simd::detail::setter_by_each_invoker<
            typename mask_type::vector_t, sizeof(bits_contain_type),
            mask_type::lane_width>;

        constexpr bits_contain_type true_val{ std::numeric_limits<bits_contain_type>::max() };
        constexpr bits_contain_type false_val{ 0 };

        alignas(alignof(typename mask_type::vector_t)) bits_contain_type temp[mask_type::bit_width];
        auto args_tuple = std::forward_as_tuple(std::forward<Args>(args)...);

        [&]<std::size_t... index>(std::index_sequence<index...>)
        {
            ((temp[index] = [&]()
            {
                if constexpr (index < sizeof...(Args))
                {
                    return static_cast<bool>(std::get<index>(args_tuple)) ? true_val : false_val;
                }
                else
                {
                    return false_val;
                }
            }()), ...);
        }(std::make_index_sequence<mask_type::lane_width>{});

        setter_invoker invoker{};
        return mask_type{
            [&] <std::size_t... Is>(std::index_sequence<Is...>)
            {
                return invoker(temp[Is]...);
            }(std::make_index_sequence<mask_type::lane_width>{})
        };
    }

    template<typename source_scalar, std::size_t input_width>
    requires(fyx::simd::is_256bits_simd_v<basic_simd<source_scalar, input_width>>)
    basic_simd<source_scalar, input_width / 2> split_low(basic_simd<source_scalar, input_width> source_vec)
    {
        return basic_simd<source_scalar, input_width / 2>{ fyx::simd::detail::split_low(source_vec.data) };
    }

    template<typename source_scalar, std::size_t input_width>
    requires(fyx::simd::is_256bits_simd_v<basic_simd<source_scalar, input_width>>)
    basic_simd<source_scalar, input_width / 2> split_high(basic_simd<source_scalar, input_width> source_vec)
    {
        return basic_simd<source_scalar, input_width / 2>{ fyx::simd::detail::split_high(source_vec.data) };
    }

    template<typename source_scalar, std::size_t input_width>
    requires(fyx::simd::is_128bits_simd_v<basic_simd<source_scalar, input_width>>)
    basic_simd<source_scalar, input_width * 2> merge(
        basic_simd<source_scalar, input_width> low, basic_simd<source_scalar, input_width> high)
    {
        return basic_simd<source_scalar, input_width * 2>{ fyx::simd::detail::merge(low.data, high.data) };
    }

    template<std::size_t index, typename simd_type> 
    requires(fyx::simd::is_basic_simd_v<simd_type> && (index <= (simd_type::lane_width - 1) && index >= 0))
    typename simd_type::scalar_t extract_single_from(simd_type input)
    {
        using scalar_type = typename simd_type::scalar_t;
        constexpr std::size_t scalar_size = sizeof(scalar_type);

        if constexpr (scalar_size == 1) { return std::bit_cast<scalar_type>(fyx::simd::detail::extract_x8<index>(input.data)); }
        else if constexpr (scalar_size == 2) { return std::bit_cast<scalar_type>(fyx::simd::detail::extract_x16<index>(input.data)); }
        else if constexpr (scalar_size == 4) { return std::bit_cast<scalar_type>(fyx::simd::detail::extract_x32<index>(input.data)); }
        else if constexpr (scalar_size == 8) { return std::bit_cast<scalar_type>(fyx::simd::detail::extract_x64<index>(input.data)); }
        else
        {
            __assume(false);
        }
    }

    template<std::size_t index, typename mask_type> 
    requires(fyx::simd::is_basic_simd_v<mask_type> && (index <= (mask_type::lane_width - 1) && index >= 0))
    bool extract_single_from(mask_type input)
    {
        constexpr std::size_t single_width_bits = mask_type::bit_width / mask_type::lane_width;
        constexpr std::size_t single_width = single_width_bits / CHAR_BIT;

        if constexpr (single_width == 1) { return static_cast<bool>(fyx::simd::detail::extract_x8<index>(input.data)); }
        else if constexpr (single_width == 2) { return static_cast<bool>(fyx::simd::detail::extract_x16<index>(input.data)); }
        else if constexpr (single_width == 4) { return static_cast<bool>(fyx::simd::detail::extract_x32<index>(input.data)); }
        else if constexpr (single_width == 8) { return static_cast<bool>(fyx::simd::detail::extract_x64<index>(input.data)); }
        else
        {
            __assume(false);
        }
    }

    using uint8x32 = basic_simd<std::uint8_t, 256>;
    using uint16x16 = basic_simd<std::uint16_t, 256>;
    using uint32x8 = basic_simd<std::uint32_t, 256>;
    using uint64x4 = basic_simd<std::uint64_t, 256>;

    using sint8x32 = basic_simd<std::int8_t, 256>;
    using sint16x16 = basic_simd<std::int16_t, 256>;
    using sint32x8 = basic_simd<std::int32_t, 256>;
    using sint64x4 = basic_simd<std::int64_t, 256>;

    using uint8x16 = basic_simd<std::uint8_t, 128>;
    using uint16x8 = basic_simd<std::uint16_t, 128>;
    using uint32x4 = basic_simd<std::uint32_t, 128>;
    using uint64x2 = basic_simd<std::uint64_t, 128>;

    using sint8x16 = basic_simd<std::int8_t, 128>;
    using sint16x8 = basic_simd<std::int16_t, 128>;
    using sint32x4 = basic_simd<std::int32_t, 128>;
    using sint64x2 = basic_simd<std::int64_t, 128>;

    using float32x8 = basic_simd<float, 256>;
    using float64x4 = basic_simd<double, 256>;
    using float32x4 = basic_simd<float, 128>;
    using float64x2 = basic_simd<double, 128>;

#if defined(_FOYE_SIMD_HAS_FP16_)
    using float16x8 = basic_simd<fy::float16, 128>;
    using float16x16 = basic_simd<fy::float16, 256>;
#endif

#if defined(_FOYE_SIMD_HAS_BF16_)
    using bfloat16x8 = basic_simd<fy::bfloat16, 128>;
    using bfloat16x16 = basic_simd<fy::bfloat16, 256>;
#endif

    using mask_8x16 = basic_mask<16, 128>;
    using mask_16x8 = basic_mask<8, 128>;
    using mask_32x4 = basic_mask<4, 128>;
    using mask_64x2 = basic_mask<2, 128>;

    using mask_8x32 = basic_mask<32, 256>;
    using mask_16x16 = basic_mask<16, 256>;
    using mask_32x8 = basic_mask<8, 256>;
    using mask_64x4 = basic_mask<4, 256>;

    template<typename simd_type>
    using mask_from_simd_t = basic_mask<simd_type::lane_width, simd_type::bit_width>;
}

#endif
