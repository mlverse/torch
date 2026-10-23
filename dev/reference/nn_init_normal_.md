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
#>  0.6004  0.9733 -0.3761  0.7712  0.1285
#>  0.4472  0.8509 -0.4042 -0.0856 -0.0889
#>  0.8860 -0.2936  0.9709  1.0159 -1.0999
#> [ CPUFloatType{3,5} ]
```
