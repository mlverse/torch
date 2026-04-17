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
#>  0.2840 -0.6332  0.6820 -0.5914  0.3444
#>  0.3852  0.7540 -0.2986 -0.8205 -0.6940
#>  0.5924  0.8060 -0.0931 -0.4640  0.1849
#> [ CPUFloatType{3,5} ]
```
