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
#>  0.8103  0.6264  0.0270  0.5624  0.5356
#>  0.5773  0.5839  0.9748  0.4866  0.8491
#>  0.3341  0.6411  0.8711  0.0732  0.1698
#> [ CPUFloatType{3,5} ]
```
