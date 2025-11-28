# Std

Std

## Usage

``` r
torch_std(self, dim, unbiased = TRUE, keepdim = FALSE)
```

## Arguments

- self:

  (Tensor) the input tensor.

- dim:

  (int or tuple of ints) the dimension or dimensions to reduce.

- unbiased:

  (bool) whether to use the unbiased estimation or not

- keepdim:

  (bool) whether the output tensor has `dim` retained or not.

## std(input, unbiased=TRUE) -\> Tensor

Returns the standard-deviation of all elements in the `input` tensor.

If `unbiased` is `FALSE`, then the standard-deviation will be calculated
via the biased estimator. Otherwise, Bessel's correction will be used.

## std(input, dim, unbiased=TRUE, keepdim=False, out=NULL) -\> Tensor

Returns the standard-deviation of each row of the `input` tensor in the
dimension `dim`. If `dim` is a list of dimensions, reduce over all of
them.

If `keepdim` is `TRUE`, the output tensor is of the same size as `input`
except in the dimension(s) `dim` where it is of size 1. Otherwise, `dim`
is squeezed (see
[`torch_squeeze`](https://torch.mlverse.org/docs/dev/reference/torch_squeeze.md)),
resulting in the output tensor having 1 (or `len(dim)`) fewer
dimension(s).

If `unbiased` is `FALSE`, then the standard-deviation will be calculated
via the biased estimator. Otherwise, Bessel's correction will be used.

## Examples

``` r
if (torch_is_installed()) {

a = torch_randn(c(1, 3))
a
torch_std(a)


a = torch_randn(c(4, 4))
a
torch_std(a, dim=1)
}
#> torch_tensor
#>  0.4641
#>  1.0876
#>  1.3680
#>  0.9970
#> [ CPUFloatType{4} ]
```
