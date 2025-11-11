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
#>  0.1473
#> -0.0580
#>  0.0838
#>  0.0129
#> -0.0853
#> -0.0109
#>  0.0416
#> -0.2036
#> -0.2299
#>  0.0973
#> -0.0005
#> -0.0505
#>  0.0079
#> -0.0763
#> -0.1080
#>  0.0201
#> -0.0478
#> -0.0849
#>  0.0948
#>  0.1981
#>  0.0175
#>  0.0737
#> -0.1341
#> -0.1473
#>  0.0349
#> -0.0058
#>  0.1228
#> -0.0745
#>  0.0833
#> -0.0666
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{100} ]
```
