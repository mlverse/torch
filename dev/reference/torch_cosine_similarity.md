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
#>  0.0823
#>  0.1015
#> -0.0009
#> -0.0187
#> -0.0061
#>  0.0548
#> -0.1538
#>  0.0366
#>  0.1345
#> -0.0365
#> -0.1171
#>  0.0562
#>  0.0374
#> -0.1761
#>  0.0774
#> -0.0749
#> -0.0037
#>  0.0330
#> -0.0024
#> -0.2094
#>  0.0776
#>  0.0461
#> -0.1463
#>  0.0591
#>  0.0657
#> -0.1112
#> -0.1009
#>  0.0521
#>  0.0766
#>  0.0317
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{100} ]
```
