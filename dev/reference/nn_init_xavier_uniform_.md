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
#>  0.1609  0.1458  0.3774  0.4995 -0.6493
#> -0.0674 -0.4849  0.1413  0.6953  0.4486
#> -0.7175  0.8631 -0.3310  0.2922  0.3126
#> [ CPUFloatType{3,5} ]
```
