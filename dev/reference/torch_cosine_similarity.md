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
#>  0.0297
#>  0.0725
#>  0.0244
#>  0.1083
#> -0.0530
#>  0.1024
#> -0.0310
#> -0.0668
#>  0.0285
#> -0.0156
#> -0.0050
#> -0.0560
#>  0.0777
#> -0.1039
#> -0.1100
#> -0.0175
#> -0.0966
#> -0.0098
#>  0.0246
#>  0.0371
#>  0.1712
#>  0.1125
#>  0.1455
#>  0.0034
#> -0.1513
#> -0.0418
#>  0.0991
#>  0.1716
#>  0.0929
#>  0.0175
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{100} ]
```
