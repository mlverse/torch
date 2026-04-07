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
#>  0.1056  0.0510  1.1421 -0.2957 -0.3485
#>  0.4922 -0.4283  0.0358  0.2085 -0.6202
#>  0.0148 -0.1345 -1.1741  1.0245  0.8518
#> [ CPUFloatType{3,5} ]
```
