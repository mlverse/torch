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
#>  0.1184  0.8455  0.6088  0.2598  0.0298
#>  0.1041  0.9106  0.4734  0.7294  0.0182
#>  0.7215  0.3967  0.0536  0.9585  0.2491
#> [ CPUFloatType{3,5} ]
```
