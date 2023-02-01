/*

MIT License

Copyright (c) 2020 Lanxiao Li

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

#pragma once
#include <ATen/cuda/CUDAContext.h>

#define CHECK_CUDA(x)                                      \
  do {                                                     \
    TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor"); \
  } while (0)

#define CHECK_CONTIGUOUS(x)                                            \
  do {                                                                 \
    TORCH_CHECK(x.is_contiguous(), #x " must ne a contiguous tensor"); \
  } while (0)

#define CHECK_IS_INT(x)                                 \
  do {                                                  \
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Int, \
                #x " must be a int tensor");            \
  } while (0)

#define CHECK_IS_FLOAT(x)                                 \
  do {                                                    \
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, \
                #x " must be a float tensor");            \
  } while (0)

#define CHECK_IS_BOOL(x)                                 \
  do {                                                   \
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Bool, \
                #x " must be a bool tensor");            \
  } while (0)
