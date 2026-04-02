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
#> -0.2508 -0.0722 -0.8345  0.3494 -0.6818
#>  0.4381 -0.8138  0.5373  0.7661  0.2645
#> -0.2448 -0.5067 -0.6238 -0.6183  0.3839
#> [ CPUFloatType{3,5} ]
```
