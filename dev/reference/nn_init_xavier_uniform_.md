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
#> -0.4736 -0.6892 -0.3555 -0.4395  0.1469
#> -0.8251  0.5169  0.1926 -0.2766  0.0588
#> -0.4625  0.1472 -0.7495 -0.6578 -0.4407
#> [ CPUFloatType{3,5} ]
```
