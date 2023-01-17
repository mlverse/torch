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

#define LANTERN_BUILD

#include "sort_vert.h"

#include "../../utils.hpp"
#include "lantern/lantern.h"
#include "utils.h"

void sort_vertices_wrapper(int b, int n, int m, const float* vertices,
                           const bool* mask, const int* num_valid, int* idx);

at::Tensor sort_vertices(at::Tensor vertices, at::Tensor mask,
                         at::Tensor num_valid) {
  CHECK_CONTIGUOUS(vertices);
  CHECK_CONTIGUOUS(mask);
  CHECK_CONTIGUOUS(num_valid);
  CHECK_CUDA(vertices);
  CHECK_CUDA(mask);
  CHECK_CUDA(num_valid);
  CHECK_IS_FLOAT(vertices);
  CHECK_IS_BOOL(mask);
  CHECK_IS_INT(num_valid);

  int b = vertices.size(0);
  int n = vertices.size(1);
  int m = vertices.size(2);
  at::Tensor idx =
      torch::zeros({b, n, MAX_NUM_VERT_IDX},
                   at::device(vertices.device()).dtype(at::ScalarType::Int));

  sort_vertices_wrapper(b, n, m, vertices.data_ptr<float>(),
                        mask.data_ptr<bool>(), num_valid.data_ptr<int>(),
                        idx.data_ptr<int>());

  return idx;
}

void* _lantern_contrib_sort_vertices(void* vertices, void* mask,
                                     void* num_valid) {
  torch::Tensor result =
      sort_vertices(from_raw::Tensor(vertices), from_raw::Tensor(mask),
                    from_raw::Tensor(num_valid));
  return make_raw::Tensor(result);
}
