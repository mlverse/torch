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
#>  0.0077
#>  0.0578
#> -0.0823
#> -0.0458
#> -0.1159
#> -0.2013
#> -0.0488
#>  0.0124
#> -0.0616
#> -0.0259
#> -0.0409
#> -0.0167
#>  0.0249
#>  0.0072
#>  0.0653
#>  0.0939
#>  0.1402
#> -0.0131
#> -0.0216
#>  0.0461
#>  0.2415
#> -0.1498
#>  0.0173
#> -0.0856
#>  0.0326
#>  0.0272
#>  0.0149
#>  0.0662
#> -0.1096
#> -0.0287
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{100} ]
```
