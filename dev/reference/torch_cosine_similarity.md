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
#>  0.0835
#> -0.0134
#>  0.0631
#> -0.0789
#>  0.0273
#> -0.0443
#>  0.0927
#> -0.0490
#> -0.0354
#>  0.0232
#>  0.0579
#> -0.0286
#>  0.1215
#>  0.0164
#> -0.0940
#> -0.0231
#>  0.0400
#>  0.0737
#> -0.0396
#> -0.1267
#>  0.1371
#>  0.0949
#> -0.0532
#> -0.0541
#> -0.1252
#> -0.0817
#> -0.0515
#>  0.1381
#> -0.0031
#>  0.0457
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{100} ]
```
