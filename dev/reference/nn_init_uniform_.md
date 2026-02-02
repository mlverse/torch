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
#>  0.6396  0.2591  0.1492  0.2788  0.9398
#>  0.8916  0.3741  0.0021  0.1781  0.6548
#>  0.6168  0.1148  0.3306  0.2589  0.7066
#> [ CPUFloatType{3,5} ]
```
