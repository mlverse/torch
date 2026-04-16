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
#> -0.0735  0.4770  0.0677  0.2671 -0.5702
#>  0.3249 -0.3651  0.0334 -0.6231  0.4622
#> -0.6055  0.5374  0.1798 -0.7855  0.0784
#> [ CPUFloatType{3,5} ]
```
