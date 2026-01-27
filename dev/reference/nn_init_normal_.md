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
#> -1.3934  0.3653  0.8740  0.1909  0.3236
#>  0.8592  0.9409  0.5967  0.0031  0.7877
#>  0.4417 -1.4986  0.1989  1.3825 -0.4587
#> [ CPUFloatType{3,5} ]
```
