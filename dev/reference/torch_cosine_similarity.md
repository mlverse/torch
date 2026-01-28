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
#>  0.0167
#> -0.0274
#>  0.2039
#> -0.0131
#>  0.0415
#>  0.1667
#> -0.0207
#>  0.0893
#> -0.0797
#>  0.1340
#>  0.1212
#>  0.1053
#>  0.0384
#> -0.0508
#> -0.0577
#>  0.1411
#>  0.0148
#>  0.1635
#>  0.0056
#> -0.0035
#>  0.1478
#> -0.0106
#>  0.0575
#>  0.1007
#> -0.1201
#>  0.0836
#> -0.0936
#> -0.0015
#>  0.0451
#> -0.1006
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{100} ]
```
