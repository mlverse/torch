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
#>  0.8198  0.2079 -0.2559 -0.6308  0.5681
#> -0.0357  0.2590 -0.7503 -0.7310 -0.5369
#> -0.2333 -0.6057 -0.3677 -0.2316  0.5493
#> [ CPUFloatType{3,5} ]
```
