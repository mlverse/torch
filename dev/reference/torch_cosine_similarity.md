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
#>  0.0469
#> -0.0857
#> -0.0737
#>  0.0355
#>  0.1173
#> -0.0199
#> -0.0073
#>  0.0024
#> -0.0950
#> -0.0142
#> -0.0737
#> -0.0306
#> -0.0049
#> -0.1103
#> -0.0305
#> -0.1870
#>  0.0501
#> -0.0978
#> -0.0654
#>  0.1099
#>  0.0808
#>  0.0728
#>  0.0333
#>  0.0680
#>  0.0001
#> -0.0780
#> -0.0632
#>  0.0184
#>  0.0229
#>  0.0222
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{100} ]
```
