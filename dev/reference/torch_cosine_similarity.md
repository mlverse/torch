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
#>  0.1066
#> -0.0278
#> -0.0817
#>  0.0702
#> -0.0121
#> -0.0315
#>  0.0120
#> -0.1061
#>  0.0223
#> -0.0927
#> -0.1011
#> -0.1194
#>  0.1770
#>  0.0488
#>  0.0302
#> -0.0263
#>  0.0174
#>  0.0753
#> -0.0249
#>  0.0508
#> -0.0892
#> -0.0601
#>  0.0319
#> -0.0642
#>  0.0103
#> -0.1130
#>  0.0448
#>  0.0360
#> -0.0500
#>  0.0969
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{100} ]
```
