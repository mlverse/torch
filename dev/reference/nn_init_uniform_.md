# Uniform initialization

Fills the input Tensor with values drawn from the uniform distribution

## Usage

``` r
nn_init_uniform_(tensor, a = 0, b = 1)
```

## Arguments

- tensor:

  an n-dimensional Tensor

- a:

  the lower bound of the uniform distribution

- b:

  the upper bound of the uniform distribution

## Examples

``` r
if (torch_is_installed()) {
w <- torch_empty(3, 5)
nn_init_uniform_(w)
}
#> torch_tensor
#>  0.6768  0.6477  0.7806  0.2942  0.6329
#>  0.9504  0.6593  0.3476  0.1003  0.2511
#>  0.2134  0.3439  0.0260  0.7964  0.5619
#> [ CPUFloatType{3,5} ]
```
