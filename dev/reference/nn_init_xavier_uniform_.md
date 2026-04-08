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
#> -0.1903  0.4734  0.0297 -0.4229 -0.0792
#>  0.5223  0.7285 -0.3054 -0.1673 -0.2466
#> -0.3859  0.8422  0.8124 -0.0260 -0.1203
#> [ CPUFloatType{3,5} ]
```
