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
#> -0.6520 -0.5369  0.1954 -0.2750  0.4100
#>  0.0953  0.5879  0.4215 -0.0657  0.4966
#> -0.2265  0.5392  0.3308  0.3188  0.2427
#> [ CPUFloatType{3,5} ]
```
