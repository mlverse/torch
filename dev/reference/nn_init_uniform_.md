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
#>  0.3232  0.1607  0.8548  0.8893  0.8216
#>  0.8854  0.3714  0.4357  0.7932  0.6518
#>  0.1794  0.7279  0.2145  0.2344  0.1227
#> [ CPUFloatType{3,5} ]
```
