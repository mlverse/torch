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
#>  0.8987 -0.3225 -0.9290 -0.3697 -1.3083
#>  2.6336 -0.2636  0.1895 -0.3766 -0.6843
#>  1.0549  0.3265  0.6227  0.4624  0.9801
#> [ CPUFloatType{3,5} ]
```
