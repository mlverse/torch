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
#> -0.4103  0.4391  0.7159 -0.0665 -0.8979
#>  0.4374  0.0722 -0.9188 -1.9536  0.4592
#> -1.9848  1.7455  1.1445  1.0526  0.9558
#> [ CPUFloatType{3,5} ]
```
