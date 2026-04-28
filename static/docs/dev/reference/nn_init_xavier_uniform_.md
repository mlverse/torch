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
#> -0.1963  0.6258 -0.4711 -0.3186 -0.1363
#> -0.6258 -0.4192  0.4723 -0.2859 -0.4944
#> -0.5350 -0.1286  0.4475 -0.2032 -0.6035
#> [ CPUFloatType{3,5} ]
```
