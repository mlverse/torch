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
#> -0.2097 -0.5539  0.6551  0.5887 -0.4482
#>  0.1579 -0.3986 -0.4793  0.2715 -0.4759
#> -0.3851 -0.4691 -0.4329  0.3357  0.8652
#> [ CPUFloatType{3,5} ]
```
