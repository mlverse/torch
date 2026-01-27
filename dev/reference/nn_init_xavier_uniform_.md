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
#> -0.6149  0.3401  0.1882 -0.0809  0.0943
#> -0.6748  0.7115  0.5434 -0.0917 -0.6925
#> -0.4338  0.6113 -0.8210  0.2608 -0.7791
#> [ CPUFloatType{3,5} ]
```
