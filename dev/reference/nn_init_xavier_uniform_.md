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
#> -0.3479 -0.2662 -0.8223  0.6697  0.8358
#>  0.3959 -0.1989  0.2335 -0.0510 -0.0568
#> -0.0692  0.4128  0.7575 -0.7809 -0.2889
#> [ CPUFloatType{3,5} ]
```
