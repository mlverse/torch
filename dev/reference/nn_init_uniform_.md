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
#>  0.9468  0.3959  0.7984  0.4124  0.9568
#>  0.6270  0.5274  0.0214  0.7492  0.5943
#>  0.7154  0.0847  0.5305  0.7206  0.8979
#> [ CPUFloatType{3,5} ]
```
