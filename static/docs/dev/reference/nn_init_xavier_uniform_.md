# Xavier uniform initialization

Fills the input `Tensor` with values according to the method described
in
`Understanding the difficulty of training deep feedforward neural networks` -
Glorot, X. & Bengio, Y. (2010), using a uniform distribution.

## Usage

``` r
nn_init_xavier_uniform_(tensor, gain = 1)
```

## Arguments

- tensor:

  an n-dimensional `Tensor`

- gain:

  an optional scaling factor

## Examples

``` r
if (torch_is_installed()) {
w <- torch_empty(3, 5)
nn_init_xavier_uniform_(w)
}
#> torch_tensor
#>  0.3734  0.0088  0.3363  0.8496 -0.7471
#>  0.4861  0.2481 -0.5762 -0.6662 -0.3284
#>  0.2954  0.1024 -0.6843  0.1423 -0.2422
#> [ CPUFloatType{3,5} ]
```
