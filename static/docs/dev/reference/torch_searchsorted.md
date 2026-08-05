# Searchsorted

Searchsorted

## Usage

``` r
torch_searchsorted(
  sorted_sequence,
  self,
  out_int32 = FALSE,
  right = FALSE,
  side = NULL,
  sorter = list()
)
```

## Arguments

- sorted_sequence:

  (Tensor) N-D or 1-D tensor, containing monotonically increasing
  sequence on the *innermost* dimension.

- self:

  (Tensor or Scalar) N-D tensor or a Scalar containing the search
  value(s).

- out_int32:

  (bool, optional) – indicate the output data type.
  [`torch_int32()`](https://torch.mlverse.org/docs/dev/reference/torch_dtype.md)
  if True,
  [`torch_int64()`](https://torch.mlverse.org/docs/dev/reference/torch_dtype.md)
  otherwise. Default value is FALSE, i.e. default output data type is
  [`torch_int64()`](https://torch.mlverse.org/docs/dev/reference/torch_dtype.md).

- right:

  (bool, optional) – if False, return the first suitable location that
  is found. If True, return the last such index. If no suitable index
  found, return 0 for non-numerical value (eg. nan, inf) or the size of
  boundaries (one pass the last index). In other words, if False, gets
  the lower bound index for each value in input from boundaries. If
  True, gets the upper bound index instead. Default value is False.

- side:

  the same as right but preferred. “left” corresponds to `FALSE` for
  right and “right” corresponds to `TRUE` for right. It will error if
  this is set to “left” while right is `TRUE`.

- sorter:

  if provided, a tensor matching the shape of the unsorted
  `sorted_sequence` containing a sequence of indices that sort it in the
  ascending order on the innermost dimension.

## searchsorted(sorted_sequence, values, \*, out_int32=FALSE, right=FALSE, out=None) -\> Tensor

Find the indices from the *innermost* dimension of `sorted_sequence`
such that, if the corresponding values in `values` were inserted before
the indices, the order of the corresponding *innermost* dimension within
`sorted_sequence` would be preserved. Return a new tensor with the same
size as `values`. If `right` is FALSE (default), then the left boundary
of `sorted_sequence` is closed.

## Examples

``` r
if (torch_is_installed()) {

sorted_sequence <- torch_tensor(rbind(c(1, 3, 5, 7, 9), c(2, 4, 6, 8, 10)))
sorted_sequence
values <- torch_tensor(rbind(c(3, 6, 9), c(3, 6, 9)))
values
torch_searchsorted(sorted_sequence, values)
torch_searchsorted(sorted_sequence, values, right=TRUE)
sorted_sequence_1d <- torch_tensor(c(1, 3, 5, 7, 9))
sorted_sequence_1d
torch_searchsorted(sorted_sequence_1d, values)
}
#> torch_tensor
#>  1  3  4
#>  1  3  4
#> [ CPULongType{2,3} ]
```
