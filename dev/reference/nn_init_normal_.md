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
#> -0.5574  1.4542 -0.8547 -1.5624 -1.1187
#>  0.5606 -1.9207 -0.6048 -0.0442 -1.2598
#>  0.1197  0.3933  0.9976  2.1448  0.8761
#> [ CPUFloatType{3,5} ]
```
