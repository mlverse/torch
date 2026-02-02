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
#>  0.9656  0.6060  0.3580  0.5171  0.3228
#>  0.2514  0.7992  0.7995  0.3930  0.2234
#>  0.2845  0.3939  0.4185  0.4925  0.4892
#> [ CPUFloatType{3,5} ]
```
