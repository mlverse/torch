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
#> -0.5031  0.1151  0.5409 -0.4913  0.8500
#> -0.0131  0.2855  0.0299 -0.5041  0.6560
#> -0.5012 -0.3909 -0.0376 -0.2795 -0.0579
#> [ CPUFloatType{3,5} ]
```
