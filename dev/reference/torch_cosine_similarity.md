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
#> -0.0568
#>  0.0560
#> -0.1223
#>  0.1001
#> -0.0276
#> -0.0363
#>  0.0839
#> -0.1081
#> -0.1009
#>  0.0166
#>  0.0323
#> -0.1262
#>  0.0026
#> -0.0848
#> -0.0579
#> -0.1332
#>  0.0337
#> -0.0059
#> -0.1149
#> -0.1303
#>  0.0299
#>  0.0543
#>  0.0358
#> -0.1209
#>  0.0243
#>  0.1084
#> -0.1661
#>  0.0687
#> -0.1421
#>  0.0576
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{100} ]
```
