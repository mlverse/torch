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
#> -0.0590
#>  0.0968
#> -0.2472
#>  0.0192
#>  0.0627
#> -0.1441
#>  0.0701
#>  0.0855
#>  0.0175
#>  0.0931
#>  0.0532
#>  0.0062
#>  0.0602
#> -0.0194
#> -0.0139
#>  0.0611
#> -0.0707
#> -0.0945
#> -0.1006
#> -0.0008
#> -0.2532
#> -0.0886
#> -0.0311
#> -0.1255
#>  0.0855
#> -0.1916
#>  0.0037
#> -0.0236
#> -0.0724
#>  0.0063
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{100} ]
```
