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
#> -1.1829  1.2166 -0.6118 -1.4883  1.7288
#>  1.1576 -1.1586  0.5230 -0.8250  1.5096
#> -0.4360 -0.1866  0.1208  0.7855 -1.0663
#> [ CPUFloatType{3,5} ]
```
