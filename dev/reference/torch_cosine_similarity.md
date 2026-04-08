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
#> -0.1243
#>  0.0001
#>  0.0293
#>  0.0384
#>  0.1694
#>  0.0264
#>  0.0063
#>  0.0766
#> -0.1074
#>  0.0622
#> -0.0649
#>  0.1161
#> -0.1308
#>  0.0222
#>  0.1171
#> -0.1909
#>  0.0851
#>  0.0091
#>  0.0220
#>  0.1017
#>  0.0956
#>  0.0691
#> -0.0027
#> -0.0246
#>  0.0005
#>  0.0938
#>  0.0832
#> -0.0130
#>  0.0066
#>  0.0046
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{100} ]
```
