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
#>  0.5188  0.8442 -0.3224 -0.1952 -0.3516
#>  0.4055 -0.5725 -0.6966  0.0238 -0.0454
#> -0.5728 -0.2612  0.2161 -0.0668  0.8282
#> [ CPUFloatType{3,5} ]
```
