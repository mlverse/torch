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
#>  0.5685  0.3814  0.0500  0.5305  0.0356
#>  0.5388  0.4222  0.5739  0.9495  0.8661
#>  0.4723  0.6234  0.5567  0.6378  0.3604
#> [ CPUFloatType{3,5} ]
```
