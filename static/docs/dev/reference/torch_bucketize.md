# Bucketize

Bucketize

## Usage

``` r
torch_bucketize(self, boundaries, out_int32 = FALSE, right = FALSE)
```

## Arguments

- self:

  (Tensor or Scalar) N-D tensor or a Scalar containing the search
  value(s).

- boundaries:

  (Tensor) 1-D tensor, must contain a monotonically increasing sequence.

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

## bucketize(input, boundaries, \*, out_int32=FALSE, right=FALSE, out=None) -\> Tensor

Returns the indices of the buckets to which each value in the `input`
belongs, where the boundaries of the buckets are set by `boundaries`.
Return a new tensor with the same size as `input`. If `right` is FALSE
(default), then the left boundary is closed.

## Examples

``` r
if (torch_is_installed()) {

boundaries <- torch_tensor(c(1, 3, 5, 7, 9))
boundaries
v <- torch_tensor(rbind(c(3, 6, 9), c(3, 6, 9)))
v
torch_bucketize(v, boundaries)
torch_bucketize(v, boundaries, right=TRUE)
}
#> torch_tensor
#>  2  3  5
#>  2  3  5
#> [ CPULongType{2,3} ]
```
