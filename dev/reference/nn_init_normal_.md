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
#> -1.8096 -0.0034 -0.0878  0.0907 -0.6675
#>  0.3330  2.1090 -0.6612  0.3226  0.8524
#> -0.9382 -1.0524  1.3524 -0.4064  0.3463
#> [ CPUFloatType{3,5} ]
```
