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
#>  0.7058  0.1407  0.3153 -1.3348  0.6603
#> -1.6723 -0.6990 -2.4278 -1.0284  0.5932
#>  1.3690 -0.3800  0.4443 -1.4694  1.6395
#> [ CPUFloatType{3,5} ]
```
