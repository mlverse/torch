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
#>  0.0327
#>  0.1283
#> -0.0241
#>  0.0313
#> -0.0354
#>  0.0336
#>  0.0057
#>  0.0234
#> -0.0398
#> -0.1488
#> -0.0833
#> -0.0261
#>  0.0177
#>  0.0424
#>  0.1517
#> -0.0829
#> -0.0180
#> -0.0836
#> -0.0352
#>  0.0209
#> -0.0824
#>  0.0313
#> -0.0228
#> -0.0231
#> -0.0713
#> -0.0060
#>  0.0777
#> -0.1571
#>  0.0464
#>  0.0053
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{100} ]
```
