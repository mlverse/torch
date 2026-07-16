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
#>  0.0819
#> -0.0372
#>  0.0125
#>  0.2488
#> -0.1662
#>  0.0848
#>  0.1424
#>  0.0028
#>  0.1831
#>  0.0989
#> -0.1660
#>  0.1260
#> -0.0240
#> -0.0902
#> -0.1371
#>  0.0146
#> -0.0565
#> -0.1226
#>  0.0326
#> -0.0874
#> -0.2193
#>  0.1237
#> -0.0445
#> -0.0243
#> -0.0526
#> -0.0960
#> -0.0083
#>  0.0542
#> -0.0808
#> -0.0461
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{100} ]
```
