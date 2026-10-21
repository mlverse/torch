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
#> -0.5936  0.2773 -0.0889  0.8440 -0.5322
#> -0.2151  0.2281  0.4146 -0.3419 -0.1091
#> -0.1065 -0.3232 -0.0053 -0.5382 -0.3573
#> [ CPUFloatType{3,5} ]
```
