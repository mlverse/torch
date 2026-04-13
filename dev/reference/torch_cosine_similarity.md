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
#>  0.0313
#> -0.0988
#>  0.0967
#>  0.1808
#>  0.0012
#>  0.1732
#>  0.0115
#> -0.0392
#> -0.0250
#>  0.0963
#> -0.1114
#>  0.1092
#> -0.1798
#> -0.0905
#>  0.0568
#> -0.0056
#> -0.0753
#>  0.0273
#>  0.1470
#>  0.0675
#>  0.0253
#> -0.0449
#> -0.0194
#> -0.0349
#>  0.0888
#> -0.0887
#>  0.0416
#>  0.0189
#>  0.0627
#>  0.0893
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{100} ]
```
