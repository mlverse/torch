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
#>  0.2966  0.2267  0.0884  0.2506 -0.0912
#>  0.1398 -0.7316  0.8138 -0.5656 -0.5243
#> -0.6298 -0.4461 -0.3093  0.6114  0.2277
#> [ CPUFloatType{3,5} ]
```
