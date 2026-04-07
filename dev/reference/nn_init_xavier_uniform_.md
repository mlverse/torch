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
#>  0.7626 -0.3836 -0.3822  0.6883  0.4990
#>  0.4115 -0.2766 -0.8258 -0.1042 -0.0591
#>  0.3067 -0.5569  0.6717  0.4327 -0.2520
#> [ CPUFloatType{3,5} ]
```
