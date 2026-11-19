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
#>  0.0334
#>  0.0419
#>  0.0266
#> -0.0797
#>  0.1072
#> -0.0634
#>  0.0702
#> -0.2036
#>  0.0896
#>  0.0554
#>  0.1217
#>  0.1345
#> -0.0114
#>  0.0629
#>  0.1493
#>  0.1522
#> -0.0043
#> -0.0565
#>  0.0104
#> -0.0086
#> -0.0124
#>  0.1215
#>  0.0397
#>  0.0614
#> -0.0501
#>  0.1594
#>  0.1182
#> -0.0168
#>  0.0594
#>  0.1200
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{100} ]
```
