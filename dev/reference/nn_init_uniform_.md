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
#>  0.9478  0.9133  0.2390  0.1258  0.0337
#>  0.7289  0.7578  0.6108  0.3836  0.5761
#>  0.3498  0.5229  0.8503  0.6839  0.3280
#> [ CPUFloatType{3,5} ]
```
