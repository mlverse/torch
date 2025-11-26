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
#> -0.0487
#> -0.0586
#> -0.0846
#>  0.0047
#> -0.0060
#> -0.0246
#>  0.0245
#>  0.0111
#> -0.0366
#> -0.0332
#>  0.0408
#> -0.0712
#> -0.0030
#>  0.1119
#> -0.1563
#>  0.0975
#>  0.0129
#> -0.1072
#>  0.0404
#>  0.0603
#>  0.0499
#>  0.0669
#>  0.0067
#>  0.2362
#>  0.0583
#>  0.1385
#>  0.0478
#> -0.0072
#>  0.1077
#>  0.0083
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{100} ]
```
