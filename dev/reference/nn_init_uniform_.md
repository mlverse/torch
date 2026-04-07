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
#>  0.6896  0.8748  0.1184  0.0989  0.1697
#>  0.9672  0.8656  0.1282  0.8593  0.7885
#>  0.6099  0.9452  0.8121  0.8746  0.9824
#> [ CPUFloatType{3,5} ]
```
