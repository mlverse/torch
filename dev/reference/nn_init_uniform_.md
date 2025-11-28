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
#>  0.2103  0.7833  0.5672  0.4093  0.7686
#>  0.2202  0.0029  0.8987  0.4905  0.0415
#>  0.4557  0.2123  0.2060  0.4704  0.1693
#> [ CPUFloatType{3,5} ]
```
