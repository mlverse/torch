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
#>  0.2614  0.1923  0.5107  0.7034  0.2094
#>  0.6667  0.0598  0.1380  0.4984 -0.4058
#>  0.1240  0.3454 -0.2685 -0.6089 -0.3558
#> [ CPUFloatType{3,5} ]
```
