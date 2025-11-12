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
#> -0.0206 -0.1198  0.3151 -0.6526  0.8159
#> -0.7697  0.8375 -0.6827 -0.4932 -0.4939
#>  0.2620  0.0413 -0.2493 -0.5166  0.3480
#> [ CPUFloatType{3,5} ]
```
