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
#> -0.0237
#> -0.1289
#>  0.0497
#>  0.0058
#> -0.0234
#>  0.0189
#>  0.1163
#>  0.0568
#>  0.0355
#> -0.0420
#>  0.0196
#> -0.0697
#> -0.0698
#> -0.1137
#>  0.0177
#> -0.0946
#> -0.1516
#>  0.1668
#> -0.0303
#>  0.0032
#>  0.0622
#> -0.0254
#>  0.0597
#> -0.0095
#> -0.0650
#> -0.0182
#> -0.0452
#> -0.0653
#> -0.1688
#> -0.0139
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{100} ]
```
