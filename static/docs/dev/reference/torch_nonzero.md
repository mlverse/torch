# Nonzero

Nonzero elements of tensors.

## Usage

``` r
torch_nonzero(self, as_list = FALSE)
```

## Arguments

- self:

  (Tensor) the input tensor.

- as_list:

  If `FALSE`, the output tensor containing indices. If `TRUE`, one 1-D
  tensor for each dimension, containing the indices of each nonzero
  element along that dimension.

  **When** `as_list` **is `FALSE` (default)**:

  Returns a tensor containing the indices of all non-zero elements of
  `input`. Each row in the result contains the indices of a non-zero
  element in `input`. The result is sorted lexicographically, with the
  last index changing the fastest (C-style).

  If `input` has \\n\\ dimensions, then the resulting indices tensor
  `out` is of size \\(z \times n)\\, where \\z\\ is the total number of
  non-zero elements in the `input` tensor.

  **When** `as_list` **is `TRUE`**:

  Returns a tuple of 1-D tensors, one for each dimension in `input`,
  each containing the indices (in that dimension) of all non-zero
  elements of `input` .

  If `input` has \\n\\ dimensions, then the resulting tuple contains
  \\n\\ tensors of size \\z\\, where \\z\\ is the total number of
  non-zero elements in the `input` tensor.

  As a special case, when `input` has zero dimensions and a nonzero
  scalar value, it is treated as a one-dimensional tensor with one
  element.

## Examples

``` r
if (torch_is_installed()) {

torch_nonzero(torch_tensor(c(1, 1, 1, 0, 1)))
}
#> torch_tensor
#>  1
#>  2
#>  3
#>  5
#> [ CPULongType{4,1} ]
```
