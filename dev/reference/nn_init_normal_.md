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
#>  0.1357  0.5241  0.1220  0.4945 -0.1169
#>  0.0350  0.0876  0.0164  0.8821 -0.1269
#>  0.9863 -0.0775  0.0965 -1.4735 -1.0717
#> [ CPUFloatType{3,5} ]
```
