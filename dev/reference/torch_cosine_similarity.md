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
#> -0.0202
#> -0.0417
#>  0.0930
#>  0.1405
#> -0.0494
#>  0.0085
#> -0.0429
#> -0.1704
#> -0.1437
#> -0.1042
#>  0.0216
#> -0.0962
#>  0.0958
#>  0.1283
#> -0.0764
#> -0.0451
#>  0.0323
#> -0.1508
#>  0.0912
#> -0.0554
#> -0.0540
#> -0.0248
#>  0.0252
#>  0.0516
#>  0.0532
#> -0.1728
#> -0.1569
#>  0.0551
#>  0.0546
#>  0.0622
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{100} ]
```
