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
#> -0.6611  0.6210 -0.0584 -0.5911 -2.1163
#>  0.6459  0.6451  1.9117 -1.3531  0.0253
#>  1.7475 -1.3604 -0.8598  0.9297  1.9689
#> [ CPUFloatType{3,5} ]
```
