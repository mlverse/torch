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
#> -0.1102
#> -0.0150
#> -0.0381
#>  0.1448
#> -0.0805
#>  0.0378
#> -0.0856
#>  0.0498
#>  0.0573
#>  0.0375
#> -0.0798
#> -0.1804
#>  0.1793
#> -0.0544
#> -0.0905
#> -0.0190
#>  0.0682
#> -0.1023
#>  0.0530
#> -0.1930
#> -0.1240
#> -0.0635
#>  0.0103
#> -0.0895
#>  0.1308
#> -0.0790
#>  0.0057
#>  0.1446
#> -0.0331
#> -0.1502
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{100} ]
```
