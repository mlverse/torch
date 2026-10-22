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
#> -0.1108
#>  0.0812
#>  0.0906
#> -0.0739
#> -0.0887
#>  0.0386
#> -0.0642
#>  0.0705
#>  0.0140
#>  0.0755
#>  0.1119
#> -0.0997
#>  0.0508
#>  0.1065
#>  0.1962
#>  0.1330
#>  0.1080
#>  0.0580
#>  0.0415
#> -0.0550
#>  0.0070
#>  0.1423
#>  0.1285
#>  0.0786
#>  0.0726
#>  0.0454
#>  0.0391
#> -0.0607
#>  0.0581
#> -0.0034
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{100} ]
```
