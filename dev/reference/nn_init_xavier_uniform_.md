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
#>  0.5454  0.4569 -0.5709  0.5282 -0.5887
#>  0.3577  0.8462  0.0732 -0.8207 -0.7612
#>  0.4184  0.4121 -0.4344  0.1459 -0.2460
#> [ CPUFloatType{3,5} ]
```
