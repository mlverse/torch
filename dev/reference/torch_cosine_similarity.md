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
#>  0.0805
#>  0.1601
#>  0.0198
#> -0.0531
#> -0.1038
#>  0.1968
#>  0.0114
#>  0.2478
#>  0.1572
#>  0.1491
#>  0.0914
#> -0.0832
#> -0.0386
#> -0.0751
#> -0.0451
#>  0.0221
#>  0.0714
#> -0.0503
#> -0.0484
#>  0.0469
#> -0.1734
#>  0.1227
#> -0.0533
#>  0.1196
#> -0.0564
#> -0.1748
#>  0.0100
#> -0.0112
#> -0.1524
#>  0.0452
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{100} ]
```
