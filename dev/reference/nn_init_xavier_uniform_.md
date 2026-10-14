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
#> -0.6293  0.0145 -0.2342 -0.7756  0.2865
#> -0.0808  0.6639 -0.8229  0.5860 -0.0261
#> -0.4421 -0.1700  0.3486 -0.4141 -0.3484
#> [ CPUFloatType{3,5} ]
```
