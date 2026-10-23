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
#>  0.5826  0.2373  0.0920  0.7258  0.2968
#>  0.1341  0.6163  0.6919  0.1232  0.7744
#>  0.5991  0.6190  0.1900  0.2649  0.8207
#> [ CPUFloatType{3,5} ]
```
