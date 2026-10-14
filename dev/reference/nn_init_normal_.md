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
#>  0.9299 -0.2313  0.1194 -1.9890 -1.4848
#> -0.1534 -0.9151 -0.3176 -0.1526  1.6269
#> -0.4617 -1.5454  0.0239 -1.6957 -0.9230
#> [ CPUFloatType{3,5} ]
```
