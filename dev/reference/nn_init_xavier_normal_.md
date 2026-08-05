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
#>  0.1040 -0.3135 -0.4022 -0.1994 -0.3395
#>  0.6933  0.0557 -0.1429  0.0936 -0.5479
#> -0.5100 -0.0916 -0.1090  0.5272  0.2864
#> [ CPUFloatType{3,5} ]
```
