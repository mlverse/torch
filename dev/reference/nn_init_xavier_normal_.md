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
#>  0.4439  0.5147 -0.0760 -0.1790  0.2368
#> -0.4058  0.0344 -0.4773 -0.5158  0.0997
#>  0.7181  0.0472 -0.1620 -0.5035 -0.6886
#> [ CPUFloatType{3,5} ]
```
