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
#> -0.1101
#>  0.1433
#> -0.0125
#> -0.0213
#>  0.0291
#> -0.1164
#> -0.0431
#>  0.0903
#> -0.0307
#> -0.0662
#> -0.0971
#>  0.0239
#> -0.1439
#> -0.0942
#> -0.0460
#>  0.0366
#>  0.0876
#> -0.1547
#> -0.1643
#> -0.0873
#>  0.0089
#> -0.0925
#>  0.0895
#>  0.0343
#>  0.0756
#> -0.0752
#>  0.0855
#>  0.0123
#>  0.0886
#> -0.0005
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{100} ]
```
