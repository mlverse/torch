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
#>  0.7909 -0.5348  0.4582  0.4465  0.3471
#> -0.0842  1.2218  0.1053  0.5133 -0.5845
#> -0.5537  0.1445  0.6265  0.5514 -0.8063
#> [ CPUFloatType{3,5} ]
```
