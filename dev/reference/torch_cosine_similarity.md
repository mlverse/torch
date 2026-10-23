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
#>  0.1224
#>  0.0305
#> -0.1647
#> -0.0081
#> -0.0314
#> -0.0013
#>  0.0843
#>  0.0373
#> -0.0218
#> -0.0895
#>  0.0269
#>  0.1324
#> -0.0789
#>  0.0490
#> -0.0333
#>  0.0132
#> -0.0864
#>  0.0319
#> -0.0018
#>  0.0825
#> -0.1073
#> -0.0741
#>  0.0295
#>  0.0472
#> -0.0355
#>  0.1286
#> -0.0091
#> -0.0362
#>  0.1386
#> -0.0477
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{100} ]
```
