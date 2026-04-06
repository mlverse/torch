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
#> -0.0124  0.8376 -0.5917  0.3275 -0.2922
#>  0.4878 -0.4995  0.5177 -0.0179  0.3967
#>  0.3647  0.0690  0.4969 -0.4081 -0.3570
#> [ CPUFloatType{3,5} ]
```
