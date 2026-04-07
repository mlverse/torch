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
#>  0.6578  0.7942 -0.0358 -0.8085 -0.1185
#>  0.6212  0.4268  0.4521 -0.3921  0.8321
#>  0.7414 -0.3247 -0.7851 -0.5720 -0.3928
#> [ CPUFloatType{3,5} ]
```
