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
#> -0.8618 -0.3340  0.2660 -0.1378 -0.2368
#>  0.2214 -0.6115 -0.5682 -0.2220 -0.4527
#> -0.0174 -0.4053 -0.0563 -0.4142  0.8128
#> [ CPUFloatType{3,5} ]
```
