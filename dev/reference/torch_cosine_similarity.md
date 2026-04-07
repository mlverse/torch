# Cosine_similarity

Cosine_similarity

## Usage

``` r
torch_cosine_similarity(x1, x2, dim = 2L, eps = 1e-08)
```

## Arguments

- x1:

  (Tensor) First input.

- x2:

  (Tensor) Second input (of size matching x1).

- dim:

  (int, optional) Dimension of vectors. Default: 1

- eps:

  (float, optional) Small value to avoid division by zero. Default: 1e-8

## cosine_similarity(x1, x2, dim=1, eps=1e-8) -\> Tensor

Returns cosine similarity between x1 and x2, computed along dim.

\$\$ \mbox{similarity} = \frac{x_1 \cdot x_2}{\max(\Vert x_1 \Vert \_2
\cdot \Vert x_2 \Vert \_2, \epsilon)} \$\$

## Examples

``` r
if (torch_is_installed()) {

input1 = torch_randn(c(100, 128))
input2 = torch_randn(c(100, 128))
output = torch_cosine_similarity(input1, input2)
output
}
#> torch_tensor
#> -0.0565
#> -0.0622
#>  0.0999
#>  0.1942
#> -0.0338
#>  0.0092
#> -0.0880
#> -0.1110
#>  0.0160
#> -0.0398
#> -0.0175
#> -0.0364
#>  0.0095
#>  0.0624
#> -0.1729
#>  0.0300
#>  0.1113
#>  0.1085
#> -0.0144
#>  0.0973
#> -0.1770
#>  0.1108
#> -0.0912
#>  0.0876
#>  0.0926
#> -0.0494
#>  0.0985
#>  0.0012
#>  0.0268
#> -0.0448
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{100} ]
```
