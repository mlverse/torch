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
#> -0.1063
#> -0.0299
#>  0.0951
#>  0.0884
#> -0.1634
#> -0.0703
#> -0.1553
#>  0.0713
#> -0.0355
#> -0.0170
#> -0.0043
#>  0.0244
#> -0.1525
#>  0.0086
#> -0.0924
#> -0.0251
#> -0.0332
#> -0.1166
#> -0.0628
#> -0.0030
#> -0.0159
#>  0.0236
#> -0.2912
#> -0.0094
#> -0.0005
#>  0.1050
#>  0.0728
#> -0.0625
#> -0.1066
#>  0.1192
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{100} ]
```
