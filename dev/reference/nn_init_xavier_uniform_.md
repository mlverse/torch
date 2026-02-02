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
#> -0.2885  0.8016  0.2671  0.6929 -0.3408
#>  0.4783  0.4668  0.3913 -0.7870  0.4415
#>  0.2813  0.7060 -0.2810 -0.0499 -0.6675
#> [ CPUFloatType{3,5} ]
```
