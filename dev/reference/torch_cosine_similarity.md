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
#>  0.0537
#>  0.0584
#> -0.0539
#>  0.0051
#>  0.0518
#>  0.0181
#> -0.0932
#>  0.0007
#> -0.0390
#> -0.0055
#>  0.0938
#>  0.0799
#>  0.0689
#>  0.0672
#>  0.0391
#>  0.0138
#> -0.1341
#>  0.0780
#>  0.0743
#> -0.0307
#>  0.1393
#>  0.0535
#> -0.0778
#>  0.1144
#>  0.0764
#>  0.0537
#> -0.0101
#>  0.1098
#> -0.0177
#>  0.2102
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{100} ]
```
