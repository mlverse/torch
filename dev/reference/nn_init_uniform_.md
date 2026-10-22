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
#>  0.1868  0.4892  0.6372  0.3977  0.4831
#>  0.7111  0.9332  0.0131  0.9553  0.2484
#>  0.0651  0.9365  0.6715  0.8412  0.8932
#> [ CPUFloatType{3,5} ]
```
