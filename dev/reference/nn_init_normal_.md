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
#>  0.7317 -1.7555 -1.0706 -1.5111  0.2399
#> -1.5089 -0.9782 -0.0458  0.7473  1.6803
#> -2.4417  0.3841 -1.1297  1.3955 -1.0957
#> [ CPUFloatType{3,5} ]
```
