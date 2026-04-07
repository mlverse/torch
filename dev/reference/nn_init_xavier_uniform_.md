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
#>  0.0419 -0.5947 -0.3651 -0.1966 -0.1196
#> -0.1844  0.4146  0.4529 -0.0616 -0.6864
#> -0.0571  0.2321 -0.0950  0.4303 -0.4464
#> [ CPUFloatType{3,5} ]
```
