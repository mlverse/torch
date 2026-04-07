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
#>  1.1831 -0.7289 -0.0280  0.1119 -1.0859
#>  0.1011 -0.6614 -0.5791  0.5624  0.8519
#> -0.9703 -0.8376 -1.9646  0.0532 -0.8777
#> [ CPUFloatType{3,5} ]
```
