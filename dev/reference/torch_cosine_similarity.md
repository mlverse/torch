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
#>  0.1065
#> -0.0163
#>  0.0969
#>  0.0149
#>  0.0548
#> -0.1310
#>  0.1256
#>  0.0819
#> -0.1072
#>  0.0436
#>  0.0120
#> -0.0069
#> -0.1651
#> -0.2119
#>  0.1715
#> -0.0134
#> -0.1406
#>  0.2142
#>  0.0028
#> -0.0869
#> -0.0302
#> -0.2025
#> -0.0503
#> -0.0185
#> -0.0331
#>  0.0786
#> -0.0427
#> -0.0747
#> -0.0450
#> -0.1277
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{100} ]
```
