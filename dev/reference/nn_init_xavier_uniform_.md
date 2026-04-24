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
#>  0.8214 -0.5134  0.7747 -0.5590  0.4351
#> -0.3155  0.7360  0.7752  0.3581 -0.0249
#> -0.2626  0.2531 -0.1707  0.8003 -0.6358
#> [ CPUFloatType{3,5} ]
```
