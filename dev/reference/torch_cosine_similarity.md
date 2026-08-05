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
#> -0.1817
#> -0.1182
#>  0.0649
#>  0.0372
#>  0.0110
#>  0.0166
#>  0.0415
#>  0.0424
#>  0.1445
#>  0.1852
#>  0.1801
#> -0.0045
#>  0.0480
#> -0.0204
#>  0.1297
#> -0.0797
#> -0.0168
#> -0.0400
#> -0.1452
#>  0.0650
#> -0.0264
#>  0.0242
#> -0.0017
#> -0.0380
#> -0.0428
#> -0.1428
#>  0.0143
#> -0.0124
#>  0.0390
#> -0.0025
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{100} ]
```
