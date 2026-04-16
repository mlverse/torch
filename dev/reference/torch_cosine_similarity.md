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
#> -0.0348
#> -0.1267
#> -0.1773
#> -0.0087
#> -0.0699
#> -0.0502
#> -0.0896
#>  0.1440
#>  0.1097
#> -0.1030
#> -0.0109
#>  0.0029
#>  0.0373
#>  0.1027
#>  0.0436
#> -0.0436
#> -0.1417
#> -0.0152
#> -0.1979
#> -0.1855
#> -0.0998
#> -0.1483
#> -0.0930
#>  0.0481
#> -0.0181
#> -0.0178
#>  0.0333
#>  0.0057
#>  0.1156
#> -0.0002
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{100} ]
```
