# Std_mean

Std_mean

## Usage

``` r
torch_std_mean(self, dim, unbiased = TRUE, keepdim = FALSE)
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

## std_mean(input, unbiased=TRUE) -\> (Tensor, Tensor)

Returns the standard-deviation and mean of all elements in the `input`
tensor.

If `unbiased` is `FALSE`, then the standard-deviation will be calculated
via the biased estimator. Otherwise, Bessel's correction will be used.

## std_mean(input, dim, unbiased=TRUE, keepdim=False) -\> (Tensor, Tensor)

Returns the standard-deviation and mean of each row of the `input`
tensor in the dimension `dim`. If `dim` is a list of dimensions, reduce
over all of them.

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
torch_std_mean(a)


a = torch_randn(c(4, 4))
a
torch_std_mean(a, 1)
}
#> [[1]]
#> torch_tensor
#>  0.9669
#>  1.3627
#>  0.9959
#>  0.2568
#> [ CPUFloatType{4} ]
#> 
#> [[2]]
#> torch_tensor
#> -0.1201
#>  0.4441
#>  0.1080
#>  0.5938
#> [ CPUFloatType{4} ]
#> 
```
