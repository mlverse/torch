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
#>  0.4028  0.6253  0.4973  0.6005  0.8442
#>  0.2230  0.5717  0.3134  0.7959  0.8652
#>  0.7304  0.6586  0.7249  0.6260  0.0130
#> [ CPUFloatType{3,5} ]
```
