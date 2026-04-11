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
#> -0.6807  0.4855  0.1751 -0.1340  0.8230
#>  0.8294  0.5862 -0.5774  0.2247 -0.7242
#> -0.4457  0.5181 -0.2803 -0.0586 -0.4050
#> [ CPUFloatType{3,5} ]
```
