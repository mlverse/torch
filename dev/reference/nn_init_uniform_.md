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
#>  0.7608  0.5157  0.6697  0.0623  0.4286
#>  0.5608  0.9889  0.3400  0.8091  0.4253
#>  0.2788  0.3689  0.0317  0.6702  0.2635
#> [ CPUFloatType{3,5} ]
```
