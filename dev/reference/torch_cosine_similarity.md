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
#> -0.0685
#>  0.1315
#>  0.0312
#> -0.0034
#>  0.1881
#>  0.0505
#> -0.1046
#>  0.0313
#> -0.0835
#> -0.0258
#> -0.0927
#> -0.0051
#>  0.0636
#>  0.0293
#> -0.0697
#>  0.0909
#>  0.1923
#>  0.0393
#> -0.0562
#> -0.0260
#>  0.0856
#>  0.0072
#> -0.0353
#> -0.0136
#> -0.0223
#> -0.0189
#>  0.0819
#>  0.0779
#>  0.0669
#> -0.1290
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{100} ]
```
