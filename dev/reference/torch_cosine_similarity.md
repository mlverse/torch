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
#>  0.0228
#> -0.0501
#> -0.0257
#>  0.0437
#>  0.1341
#> -0.0420
#>  0.0989
#>  0.0431
#>  0.0662
#> -0.0606
#> -0.0266
#> -0.0727
#> -0.0389
#>  0.0355
#>  0.0019
#> -0.0033
#>  0.0870
#>  0.0773
#> -0.0535
#> -0.0856
#>  0.1461
#> -0.0505
#> -0.1219
#>  0.0727
#> -0.1108
#>  0.0331
#> -0.0663
#>  0.0825
#>  0.0148
#>  0.0315
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{100} ]
```
