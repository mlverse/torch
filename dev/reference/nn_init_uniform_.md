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
#>  0.1802  0.9354  0.5348  0.1699  0.7993
#>  0.1386  0.5427  0.4788  0.1882  0.0096
#>  0.5258  0.6207  0.6023  0.2430  0.6851
#> [ CPUFloatType{3,5} ]
```
