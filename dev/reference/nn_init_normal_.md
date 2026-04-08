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
#>  1.7024  0.8825  0.2257 -0.1910 -0.4969
#> -2.6232 -0.7312  2.2130 -0.9439 -0.3631
#>  0.0547 -0.3437 -0.4203 -0.8441 -1.6498
#> [ CPUFloatType{3,5} ]
```
