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
#>  0.0901  0.6211  0.0716  0.2131  0.1734
#>  0.7132  0.0877  0.4778  0.6006  0.9865
#>  0.9662  0.0385  0.1028  0.9104  0.8731
#> [ CPUFloatType{3,5} ]
```
