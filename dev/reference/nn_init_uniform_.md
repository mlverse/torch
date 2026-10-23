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
#>  0.6803  0.5436  0.0956  0.5112  0.9958
#>  0.2370  0.4413  0.3743  0.6380  0.1158
#>  0.5534  0.2862  0.2194  0.2408  0.3105
#> [ CPUFloatType{3,5} ]
```
