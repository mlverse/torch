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
#>  0.0903
#> -0.0059
#> -0.0622
#>  0.1372
#>  0.1733
#>  0.0156
#> -0.1660
#>  0.0452
#>  0.1701
#>  0.0505
#> -0.1040
#>  0.0119
#> -0.0668
#>  0.0168
#>  0.0546
#>  0.1177
#>  0.1603
#>  0.0943
#>  0.1131
#> -0.0118
#> -0.0349
#> -0.0900
#> -0.0374
#>  0.0209
#>  0.0035
#>  0.0241
#> -0.1953
#> -0.0144
#> -0.1337
#>  0.0330
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{100} ]
```
