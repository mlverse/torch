# Nansum

Nansum

## Usage

``` r
torch_nansum(self, dim = NULL, keepdim = FALSE, dtype = NULL)
```

## Arguments

- self:

  (Tensor) the input tensor.

- dim:

  (int or tuple of ints) the dimension or dimensions to reduce.

- keepdim:

  (bool) whether the output tensor has `dim` retained or not.

- dtype:

  the desired data type of returned tensor. If specified, the input
  tensor is casted to dtype before the operation is performed. This is
  useful for preventing data type overflows. Default: `NULL`.

## nansum(input, \*, dtype=None) -\> Tensor

Returns the sum of all elements, treating Not a Numbers (NaNs) as zero.

## nansum(input, dim, keepdim=FALSE, \*, dtype=None) -\> Tensor

Returns the sum of each row of the `input` tensor in the given dimension
`dim`, treating Not a Numbers (NaNs) as zero. If `dim` is a list of
dimensions, reduce over all of them.

If `keepdim` is `TRUE`, the output tensor is of the same size as `input`
except in the dimension(s) `dim` where it is of size 1. Otherwise, `dim`
is squeezed (see
[`torch_squeeze`](https://torch.mlverse.org/docs/dev/reference/torch_squeeze.md)),
resulting in the output tensor having 1 (or `len(dim)`) fewer
dimension(s).

## Examples

``` r
if (torch_is_installed()) {

a <- torch_tensor(c(1., 2., NaN, 4.))
torch_nansum(a)


torch_nansum(torch_tensor(c(1., NaN)))
a <- torch_tensor(rbind(c(1, 2), c(3., NaN)))
torch_nansum(a)
torch_nansum(a, dim=1)
torch_nansum(a, dim=2)
}
#> torch_tensor
#>  3
#>  3
#> [ CPUFloatType{2} ]
```
