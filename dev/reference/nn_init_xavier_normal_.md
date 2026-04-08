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
#>  0.1188  0.7188  0.6935  0.3294 -0.2907
#>  0.6036  0.3766  0.8588  0.5395 -1.0128
#> -0.9037 -0.3087  0.7730  0.4017  1.1140
#> [ CPUFloatType{3,5} ]
```
