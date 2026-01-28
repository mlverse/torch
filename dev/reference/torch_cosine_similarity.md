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
#>  0.0658
#> -0.1412
#> -0.0820
#> -0.0620
#>  0.0633
#>  0.0775
#>  0.0212
#> -0.0923
#>  0.1463
#>  0.0569
#>  0.1552
#>  0.0817
#> -0.1590
#> -0.1896
#> -0.0403
#>  0.0416
#>  0.0340
#> -0.0394
#>  0.0061
#>  0.0930
#>  0.0037
#> -0.0792
#> -0.0310
#>  0.0065
#> -0.1424
#> -0.0261
#>  0.0502
#>  0.0449
#>  0.0040
#> -0.1199
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{100} ]
```
