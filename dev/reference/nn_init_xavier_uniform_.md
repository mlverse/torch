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
#> -0.5764  0.0486 -0.3320  0.3677 -0.6878
#> -0.4928  0.5866  0.0479  0.7096  0.5199
#> -0.1244  0.4823 -0.1651  0.2560  0.2454
#> [ CPUFloatType{3,5} ]
```
