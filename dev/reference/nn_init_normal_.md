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
#>  0.8470  0.4898 -2.5534 -1.2155  0.2838
#> -0.0165  0.3212 -0.1708  1.5334  0.4340
#> -0.8028  0.7557  1.5952  0.1680 -0.7765
#> [ CPUFloatType{3,5} ]
```
