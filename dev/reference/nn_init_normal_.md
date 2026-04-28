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
#>  0.3901  0.5186  0.4585 -1.9384 -0.9250
#> -0.0606  2.2232  1.6590  0.5676 -0.0494
#>  0.2760 -0.8774 -1.1152 -0.2767  1.0902
#> [ CPUFloatType{3,5} ]
```
