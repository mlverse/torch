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
#> -0.1264 -0.3675 -1.0165 -1.9357 -1.4429
#> -0.6419  1.4941  1.3295 -0.4597 -1.1712
#> -0.9531  0.0794  2.1274  0.1569 -0.4471
#> [ CPUFloatType{3,5} ]
```
