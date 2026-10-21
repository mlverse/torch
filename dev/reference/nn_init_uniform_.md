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
#>  0.4762  0.8018  0.8124  0.1341  0.5597
#>  0.3546  0.6121  0.5694  0.1632  0.3978
#>  0.2503  0.8976  0.4678  0.4880  0.0345
#> [ CPUFloatType{3,5} ]
```
