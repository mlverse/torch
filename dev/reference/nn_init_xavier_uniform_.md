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
#>  0.0791  0.1561 -0.2058 -0.1675 -0.6143
#>  0.2209  0.0367 -0.1074 -0.2964  0.1571
#>  0.4832 -0.1938 -0.8190  0.6779  0.1161
#> [ CPUFloatType{3,5} ]
```
