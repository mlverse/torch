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
#>  0.0498
#> -0.0236
#> -0.0951
#>  0.1445
#>  0.1231
#> -0.0633
#>  0.1278
#>  0.0279
#>  0.1433
#> -0.2646
#>  0.0119
#>  0.1296
#> -0.0096
#>  0.0045
#>  0.1149
#> -0.0232
#>  0.0248
#> -0.0821
#>  0.0422
#>  0.0366
#> -0.1590
#>  0.0684
#>  0.0005
#> -0.0924
#>  0.0719
#> -0.0314
#>  0.1033
#> -0.0032
#> -0.0543
#> -0.0641
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{100} ]
```
