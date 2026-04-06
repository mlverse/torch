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
#>  0.3193  0.3410  0.8073  0.6014  0.7617
#>  0.4040  0.4080  0.8121  0.6208  0.6785
#>  0.8132  0.9384  0.6377  0.4029  0.6656
#> [ CPUFloatType{3,5} ]
```
