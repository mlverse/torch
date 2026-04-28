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
#>  0.4589  0.4756  0.6656  0.1407  0.2925
#>  0.3768 -0.2694  0.3720  0.4403 -0.4447
#> -0.1427  0.1044 -0.5191  0.1971  0.7739
#> [ CPUFloatType{3,5} ]
```
