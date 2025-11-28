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
#>  1.2776  0.2925  0.9642  0.2920 -0.1676
#> -2.5601 -2.4764  0.9200  0.4931  0.2004
#> -0.3845 -0.0062  1.6398 -0.8176 -2.6718
#> [ CPUFloatType{3,5} ]
```
