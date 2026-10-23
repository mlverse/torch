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
#> -0.5348  0.3593 -0.3155 -0.8486  0.0068
#> -0.1175  0.1275  0.4085 -1.4018 -0.3685
#> -0.6050  0.8337 -0.4382  0.5691  0.5143
#> [ CPUFloatType{3,5} ]
```
