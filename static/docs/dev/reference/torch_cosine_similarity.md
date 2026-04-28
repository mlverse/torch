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
#>  0.0065
#>  0.1449
#>  0.1412
#>  0.1311
#> -0.1727
#>  0.1064
#>  0.1792
#>  0.0446
#> -0.0201
#> -0.0629
#>  0.1670
#>  0.0301
#> -0.0829
#> -0.1224
#>  0.0256
#>  0.0651
#> -0.0286
#>  0.0125
#>  0.0793
#> -0.0248
#>  0.0050
#>  0.0419
#> -0.0857
#>  0.0677
#> -0.0726
#> -0.1083
#>  0.0006
#> -0.0792
#>  0.0490
#> -0.0757
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{100} ]
```
