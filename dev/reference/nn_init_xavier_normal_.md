# Xavier normal initialization

Fills the input `Tensor` with values according to the method described
in
`Understanding the difficulty of training deep feedforward neural networks` -
Glorot, X. & Bengio, Y. (2010), using a normal distribution.

## Usage

``` r
nn_init_xavier_normal_(tensor, gain = 1)
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
nn_init_xavier_normal_(w)
}
#> torch_tensor
#>  0.8158 -0.2789  0.8164 -0.3136  0.4814
#>  0.5475  0.4795  0.6288  0.3028 -1.0400
#>  0.5191  0.6692  0.8152 -0.1554 -0.5655
#> [ CPUFloatType{3,5} ]
```
