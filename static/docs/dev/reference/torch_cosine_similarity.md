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
#> -0.1363
#>  0.0804
#> -0.1090
#> -0.0069
#> -0.0397
#> -0.0710
#> -0.1747
#>  0.0038
#> -0.0095
#>  0.0638
#> -0.2460
#> -0.0606
#>  0.0330
#>  0.0285
#> -0.0141
#> -0.0642
#>  0.0887
#> -0.0431
#> -0.0136
#>  0.0823
#>  0.1248
#>  0.0497
#>  0.0783
#> -0.1893
#>  0.1155
#> -0.0182
#> -0.0629
#> -0.0635
#>  0.0458
#>  0.0606
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{100} ]
```
