# Var_mean

Var_mean

## Usage

``` r
torch_var_mean(self, dim, unbiased = TRUE, keepdim = FALSE)
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

## var_mean(input, unbiased=TRUE) -\> (Tensor, Tensor)

Returns the variance and mean of all elements in the `input` tensor.

If `unbiased` is `FALSE`, then the variance will be calculated via the
biased estimator. Otherwise, Bessel's correction will be used.

## var_mean(input, dim, keepdim=False, unbiased=TRUE) -\> (Tensor, Tensor)

Returns the variance and mean of each row of the `input` tensor in the
given dimension `dim`.

If `keepdim` is `TRUE`, the output tensor is of the same size as `input`
except in the dimension(s) `dim` where it is of size 1. Otherwise, `dim`
is squeezed (see
[`torch_squeeze`](https://torch.mlverse.org/docs/dev/reference/torch_squeeze.md)),
resulting in the output tensor having 1 (or `len(dim)`) fewer
dimension(s).

If `unbiased` is `FALSE`, then the variance will be calculated via the
biased estimator. Otherwise, Bessel's correction will be used.

## Examples

``` r
if (torch_is_installed()) {

a = torch_randn(c(1, 3))
a
torch_var_mean(a)


a = torch_randn(c(4, 4))
a
torch_var_mean(a, 1)
}
#> [[1]]
#> torch_tensor
#>  0.8860
#>  0.3965
#>  1.5414
#>  0.3045
#> [ CPUFloatType{4} ]
#> 
#> [[2]]
#> torch_tensor
#> -0.1691
#> -0.3097
#>  0.6197
#>  0.2003
#> [ CPUFloatType{4} ]
#> 
```
