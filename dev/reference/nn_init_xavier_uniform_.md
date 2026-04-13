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
#> -0.3139  0.1979  0.2696 -0.0694  0.7126
#> -0.3162 -0.0775  0.5365  0.0018 -0.2875
#> -0.0506  0.6674  0.8635  0.6417 -0.3207
#> [ CPUFloatType{3,5} ]
```
