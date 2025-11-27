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
#> -0.0300  0.4923 -0.2370  0.7325 -0.3113
#> -0.0210  0.8566  0.9996 -1.0033 -0.3222
#> -0.1494 -0.1805  0.4212  0.7043 -0.2926
#> [ CPUFloatType{3,5} ]
```
