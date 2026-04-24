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
#>  0.0285
#>  0.1389
#> -0.0850
#>  0.0983
#>  0.0327
#> -0.0773
#>  0.1332
#> -0.0408
#>  0.0632
#> -0.0936
#> -0.0734
#>  0.0348
#>  0.0403
#> -0.0133
#> -0.0868
#>  0.0083
#> -0.0733
#>  0.0748
#> -0.0094
#> -0.1292
#>  0.0128
#> -0.0425
#> -0.1702
#>  0.0185
#>  0.0367
#>  0.1323
#> -0.1497
#> -0.0232
#> -0.0318
#>  0.0630
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{100} ]
```
