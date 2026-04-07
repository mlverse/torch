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
#> -0.0598
#> -0.0206
#> -0.0054
#> -0.0431
#> -0.0275
#>  0.0736
#> -0.2350
#> -0.0643
#>  0.0203
#>  0.1632
#> -0.0489
#>  0.0321
#> -0.1182
#>  0.0460
#>  0.0490
#> -0.0532
#> -0.0592
#>  0.0440
#> -0.0215
#>  0.0711
#> -0.1341
#>  0.0650
#> -0.1046
#>  0.0893
#> -0.0671
#>  0.0201
#> -0.0159
#> -0.0718
#>  0.0170
#>  0.1552
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{100} ]
```
