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
#>  0.2868 -1.0510  0.6753  0.5441  0.8381
#>  2.6772  0.3183 -1.4848 -1.1999 -0.1626
#> -0.5313 -0.8114  0.3897  1.0312  0.9681
#> [ CPUFloatType{3,5} ]
```
