# Orthogonal initialization

Fills the input `Tensor` with a (semi) orthogonal matrix, as described
in
`Exact solutions to the nonlinear dynamics of learning in deep linear neural networks` -
Saxe, A. et al. (2013). The input tensor must have at least 2
dimensions, and for tensors with more than 2 dimensions the trailing
dimensions are flattened.

## Usage

``` r
nn_init_orthogonal_(tensor, gain = 1)
```

## Arguments

- tensor:

  an n-dimensional `Tensor`

- gain:

  optional scaling factor

## Examples

``` r
if (torch_is_installed()) {
w <- torch_empty(3, 5)
nn_init_orthogonal_(w)
}
#> torch_tensor
#> -0.0061 -0.6102 -0.4794  0.5903 -0.2223
#> -0.4641 -0.6086  0.6192 -0.0705  0.1606
#>  0.8416 -0.4133  0.1395 -0.2178  0.2322
#> [ CPUFloatType{3,5} ]
```
