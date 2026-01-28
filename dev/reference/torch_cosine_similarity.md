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
#>  0.0728
#>  0.0038
#> -0.1965
#> -0.0424
#>  0.0933
#>  0.0883
#> -0.1019
#> -0.1032
#> -0.1888
#>  0.0137
#> -0.0062
#>  0.1204
#> -0.0064
#>  0.0373
#>  0.0534
#>  0.0782
#>  0.1075
#> -0.0132
#> -0.0012
#>  0.0371
#>  0.0256
#> -0.1199
#> -0.0226
#> -0.0082
#>  0.0800
#> -0.1420
#>  0.0910
#> -0.1254
#> -0.0222
#> -0.0622
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{100} ]
```
