# Sum

Sum

## Usage

``` r
torch_sum(self, dim, keepdim = FALSE, dtype = NULL)
```

## Arguments

- self:

  (Tensor) the input tensor.

- dim:

  (int or tuple of ints) the dimension or dimensions to reduce.

- keepdim:

  (bool) whether the output tensor has `dim` retained or not.

- dtype:

  (`torch.dtype`, optional) the desired data type of returned tensor. If
  specified, the input tensor is casted to `dtype` before the operation
  is performed. This is useful for preventing data type overflows.
  Default: NULL.

## sum(input, dtype=NULL) -\> Tensor

Returns the sum of all elements in the `input` tensor.

## sum(input, dim, keepdim=False, dtype=NULL) -\> Tensor

Returns the sum of each row of the `input` tensor in the given dimension
`dim`. If `dim` is a list of dimensions, reduce over all of them.

If `keepdim` is `TRUE`, the output tensor is of the same size as `input`
except in the dimension(s) `dim` where it is of size 1. Otherwise, `dim`
is squeezed (see
[`torch_squeeze`](https://torch.mlverse.org/docs/dev/reference/torch_squeeze.md)),
resulting in the output tensor having 1 (or `len(dim)`) fewer
dimension(s).

## Examples

``` r
if (torch_is_installed()) {

a = torch_randn(c(1, 3))
a
torch_sum(a)


a <- torch_randn(c(4, 4))
a
torch_sum(a, 1)
b <- torch_arange(1, 4 * 5 * 6)$view(c(4, 5, 6))
torch_sum(b, list(2, 1))
}
#> torch_tensor
#>  1160
#>  1180
#>  1200
#>  1220
#>  1240
#>  1260
#> [ CPUFloatType{6} ]
```
