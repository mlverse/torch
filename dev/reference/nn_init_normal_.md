# Normal initialization

Fills the input Tensor with values drawn from the normal distribution

## Usage

``` r
nn_init_normal_(tensor, mean = 0, std = 1)
```

## Arguments

- tensor:

  an n-dimensional Tensor

- mean:

  the mean of the normal distribution

- std:

  the standard deviation of the normal distribution

## Examples

``` r
if (torch_is_installed()) {
w <- torch_empty(3, 5)
nn_init_normal_(w)
}
#> torch_tensor
#>  0.5223 -1.8461 -0.3191  0.0084 -0.3769
#>  1.7510  0.5295 -1.5081  0.0811 -0.8488
#>  0.2407  0.8957  0.1508  2.4118 -0.0256
#> [ CPUFloatType{3,5} ]
```
