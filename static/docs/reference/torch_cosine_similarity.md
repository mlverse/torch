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
#> -0.1972
#>  0.0474
#>  0.0344
#>  0.0399
#> -0.1622
#> -0.0410
#> -0.1350
#>  0.0069
#>  0.0139
#> -0.0225
#>  0.0668
#>  0.0096
#> -0.0960
#>  0.0762
#>  0.1517
#>  0.0246
#>  0.0718
#> -0.1148
#>  0.0755
#> -0.0160
#>  0.1074
#> -0.0354
#>  0.0192
#>  0.0605
#> -0.0042
#>  0.0125
#> -0.0569
#>  0.0803
#>  0.0709
#> -0.1291
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{100} ]
```
