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
#> -0.6875  0.3962 -0.7502 -0.8188  0.4385
#>  0.5459  0.0287 -0.8548 -0.2790 -0.0145
#>  0.6822  0.3944 -0.1420 -0.0315  0.4750
#> [ CPUFloatType{3,5} ]
```
