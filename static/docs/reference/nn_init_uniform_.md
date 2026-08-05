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
#>  0.1932  0.1987  0.7498  0.0026  0.9608
#>  0.8952  0.0136  0.1753  0.4718  0.9884
#>  0.9517  0.2264  0.3119  0.3943  0.3621
#> [ CPUFloatType{3,5} ]
```
