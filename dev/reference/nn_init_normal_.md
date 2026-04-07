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
#> -1.4548  0.9908 -0.6720  0.8820 -0.2578
#> -1.3745 -1.2085 -0.6046 -1.5794  0.2800
#> -0.8998  0.7763  0.6686  0.5863  2.1123
#> [ CPUFloatType{3,5} ]
```
