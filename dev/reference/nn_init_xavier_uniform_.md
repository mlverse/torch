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
#> -0.3157 -0.8487 -0.2210 -0.4368 -0.3164
#>  0.6118 -0.0418 -0.6823 -0.4844  0.5438
#>  0.1036 -0.8567  0.7481 -0.7408 -0.4766
#> [ CPUFloatType{3,5} ]
```
