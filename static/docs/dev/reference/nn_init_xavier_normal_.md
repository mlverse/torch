# Xavier normal initialization

Fills the input `Tensor` with values according to the method described
in
`Understanding the difficulty of training deep feedforward neural networks` -
Glorot, X. & Bengio, Y. (2010), using a normal distribution.

## Usage

``` r
nn_init_xavier_normal_(tensor, gain = 1)
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
nn_init_xavier_normal_(w)
}
#> torch_tensor
#> -0.2723 -0.1876 -0.3363 -0.7351 -0.2972
#> -1.1700  0.1549 -0.3935 -0.8533 -0.4411
#> -0.6783  0.1306 -1.2685  0.2587 -0.8010
#> [ CPUFloatType{3,5} ]
```
