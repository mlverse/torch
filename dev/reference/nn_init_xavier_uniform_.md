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
#> -0.6762 -0.0999 -0.0948 -0.4429  0.8036
#> -0.6667  0.0269  0.2093  0.0217  0.6040
#> -0.3888 -0.0574  0.0758  0.8362 -0.0573
#> [ CPUFloatType{3,5} ]
```
