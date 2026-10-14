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
#>  0.0483
#>  0.0602
#> -0.1084
#> -0.0041
#> -0.0855
#> -0.0107
#> -0.0031
#>  0.2407
#> -0.0531
#> -0.0504
#>  0.0663
#>  0.0097
#>  0.0443
#>  0.1577
#>  0.0214
#>  0.1387
#> -0.1471
#> -0.0412
#> -0.0126
#> -0.0475
#>  0.0510
#>  0.1352
#> -0.0843
#>  0.0948
#>  0.0010
#> -0.1098
#> -0.0304
#>  0.1376
#>  0.1132
#>  0.0446
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{100} ]
```
