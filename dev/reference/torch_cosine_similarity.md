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
#> -0.0853
#> -0.0856
#> -0.0579
#>  0.0025
#> -0.0758
#> -0.0167
#>  0.1177
#>  0.0815
#>  0.0407
#>  0.1335
#>  0.0096
#> -0.0157
#>  0.0078
#>  0.0766
#> -0.0231
#> -0.1222
#> -0.0387
#>  0.0232
#>  0.0376
#>  0.0149
#>  0.0330
#> -0.0437
#> -0.2592
#>  0.0824
#>  0.0670
#>  0.0204
#>  0.0371
#>  0.0476
#> -0.0889
#>  0.1362
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{100} ]
```
