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
#>  0.0068
#> -0.0250
#> -0.0306
#> -0.0783
#>  0.0852
#> -0.0171
#> -0.0661
#>  0.1248
#>  0.0533
#>  0.0243
#> -0.1892
#> -0.0566
#>  0.0035
#>  0.0093
#> -0.1236
#> -0.0742
#>  0.0758
#>  0.1431
#> -0.0460
#> -0.0994
#>  0.1261
#> -0.0089
#> -0.0242
#>  0.0350
#> -0.2761
#>  0.0570
#>  0.0032
#>  0.0928
#> -0.0611
#> -0.0906
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{100} ]
```
