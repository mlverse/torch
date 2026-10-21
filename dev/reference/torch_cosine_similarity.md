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
#> -0.0014
#> -0.0355
#>  0.0327
#>  0.0330
#>  0.1361
#>  0.0033
#> -0.0706
#> -0.0056
#>  0.0237
#> -0.0502
#>  0.0017
#> -0.1220
#> -0.1587
#>  0.0220
#> -0.0276
#>  0.0120
#>  0.0060
#>  0.0068
#> -0.1199
#>  0.1561
#> -0.0517
#>  0.0138
#> -0.1728
#>  0.1510
#> -0.0635
#> -0.0237
#>  0.1398
#> -0.0216
#> -0.0667
#>  0.0404
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{100} ]
```
