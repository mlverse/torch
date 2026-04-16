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
#>  0.0361
#> -0.0370
#>  0.0602
#>  0.0790
#> -0.0073
#>  0.0510
#> -0.1104
#>  0.0899
#> -0.0343
#> -0.1390
#>  0.0461
#>  0.0330
#> -0.0399
#> -0.0186
#> -0.0603
#> -0.0002
#>  0.1655
#>  0.0490
#>  0.0350
#> -0.0885
#>  0.1847
#> -0.1045
#> -0.0830
#> -0.0495
#>  0.0245
#>  0.0484
#> -0.0886
#> -0.0478
#>  0.0088
#>  0.0071
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{100} ]
```
