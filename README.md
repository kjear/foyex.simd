[English](README.md) | [中文](README_zh.md)
Purely for fun, I write code purely as a hobby, similar in nature to playing CS and building LEGO. I want to try sending the code to attract the attention of experts, give me guidance, and even help me develop this project. I personally do not have any educational background in science, engineering, or computer science. I come from a background in humanities, painting, and art, without a strong foundation in mathematics or the practical application scenarios that programmers often encounter, so there must be a lot of problems with the code

The mandatory requirement is MSVC C++20 for the Windows platform, as it extensively utilizes the unique APIs of the Windows platform and the syntax features of C++20
There are so many mathematical functions in simd_floating.hpp that I have written using other parts of this library, which can serve as examples of using this library.
Considering that the two main vector types in simd_def.hpp are still in the process of improvement, the code implemented by these mathematical functions can become more elegant.

TODO:
  1. Improve the functions related to floating-point operations                              <- this is what i'm doing now
  2. Add support for complex numbers
  3. Add support for AVX512
  4. Add support for ARM
  5. Remove the dependency on the unique API of the MSVC compiler for the Windows platform   <- I think this is quite important
  6. Optimize the performance of existing functions implemented by composite instructions
