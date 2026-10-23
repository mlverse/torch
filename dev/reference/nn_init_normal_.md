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
#>  0.4651  0.6221  0.1592 -1.2329  0.2694
#> -0.2147 -0.4604 -1.9476 -0.3628  2.2815
#> -0.2408 -0.1895 -0.7861 -0.0634 -0.1217
#> [ CPUFloatType{3,5} ]
```
