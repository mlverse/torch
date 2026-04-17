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
#>  0.4384  2.6665 -0.6435  0.5470  0.1132
#>  0.9747  0.5887 -0.4841 -0.4427 -0.6309
#>  0.6355  1.2275 -0.5086 -0.2661  0.5046
#> [ CPUFloatType{3,5} ]
```
