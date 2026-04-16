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
#>  1.0711  1.0703  0.0947  1.1207  0.9978
#> -1.8000  0.6723 -0.5692 -0.2294  0.9008
#>  0.4315  1.2870 -0.3783  0.2942  0.1629
#> [ CPUFloatType{3,5} ]
```
