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
#> -0.0153
#> -0.0380
#> -0.1209
#> -0.0766
#>  0.0538
#> -0.1359
#> -0.0256
#> -0.0800
#> -0.1903
#> -0.0313
#> -0.0952
#> -0.0557
#>  0.1432
#> -0.0092
#> -0.0434
#> -0.0134
#>  0.0875
#> -0.0366
#> -0.0288
#>  0.1528
#>  0.0303
#>  0.0210
#>  0.0100
#> -0.0079
#> -0.0594
#> -0.0568
#>  0.1290
#> -0.0599
#>  0.0191
#>  0.0608
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{100} ]
```
