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
#> -0.1185 -0.8324  0.8585 -0.4126 -0.6787
#>  0.7900 -0.8095  0.5114 -0.6799  0.5342
#>  0.8226 -0.4119 -0.8512  0.3731  0.8484
#> [ CPUFloatType{3,5} ]
```
