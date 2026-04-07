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
#>  0.2145
#>  0.0163
#> -0.0302
#> -0.1009
#> -0.0966
#>  0.1922
#>  0.2096
#>  0.1095
#> -0.0360
#>  0.1944
#> -0.0493
#>  0.1623
#> -0.0501
#> -0.0786
#> -0.0515
#> -0.1291
#>  0.1802
#> -0.1368
#> -0.0383
#>  0.1254
#>  0.0182
#> -0.1191
#> -0.1127
#> -0.0298
#>  0.0351
#> -0.0685
#> -0.0860
#> -0.1370
#>  0.0717
#>  0.0243
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{100} ]
```
