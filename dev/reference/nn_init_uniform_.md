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
#>  0.1408  0.0891  0.9802  0.6656  0.9487
#>  0.0554  0.6953  0.4712  0.0152  0.7086
#>  0.1475  0.0435  0.4834  0.4431  0.6925
#> [ CPUFloatType{3,5} ]
```
