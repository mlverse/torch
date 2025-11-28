# Prod

Prod

## Usage

``` r
torch_prod(self, dim, keepdim = FALSE, dtype = NULL)
```

## Arguments

- self:

  (Tensor) the input tensor.

- dim:

  (int) the dimension to reduce.

- keepdim:

  (bool) whether the output tensor has `dim` retained or not.

- dtype:

  (`torch.dtype`, optional) the desired data type of returned tensor. If
  specified, the input tensor is casted to `dtype` before the operation
  is performed. This is useful for preventing data type overflows.
  Default: NULL.

## prod(input, dtype=NULL) -\> Tensor

Returns the product of all elements in the `input` tensor.

## prod(input, dim, keepdim=False, dtype=NULL) -\> Tensor

Returns the product of each row of the `input` tensor in the given
dimension `dim`.

If `keepdim` is `TRUE`, the output tensor is of the same size as `input`
except in the dimension `dim` where it is of size 1. Otherwise, `dim` is
squeezed (see
[`torch_squeeze`](https://torch.mlverse.org/docs/dev/reference/torch_squeeze.md)),
resulting in the output tensor having 1 fewer dimension than `input`.

## Examples

``` r
if (torch_is_installed()) {

a = torch_randn(c(1, 3))
a
torch_prod(a)


a = torch_randn(c(4, 2))
a
torch_prod(a, 1)
}
#> torch_tensor
#>  0.3621
#> -0.2006
#> [ CPUFloatType{2} ]
```
