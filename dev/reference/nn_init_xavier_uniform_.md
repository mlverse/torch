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
#>  0.1672  0.5602 -0.4577  0.5413  0.7536
#>  0.2807 -0.7402 -0.3860 -0.4459 -0.0483
#> -0.1857  0.3185 -0.5084 -0.8546 -0.6098
#> [ CPUFloatType{3,5} ]
```
