# Uniform initialization

Fills the input Tensor with values drawn from the uniform distribution

## Usage

``` r
nn_init_uniform_(tensor, a = 0, b = 1)
```

## Arguments

- tensor:

  an n-dimensional Tensor

- a:

  the lower bound of the uniform distribution

- b:

  the upper bound of the uniform distribution

## Examples

``` r
if (torch_is_installed()) {
w <- torch_empty(3, 5)
nn_init_uniform_(w)
}
#> torch_tensor
#>  0.6683  0.2020  0.8943  0.0630  0.4404
#>  0.1368  0.9338  0.7767  0.7916  0.6418
#>  0.5001  0.4436  0.6201  0.4128  0.6097
#> [ CPUFloatType{3,5} ]
```
