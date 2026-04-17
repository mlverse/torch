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
#> -0.1480
#>  0.1649
#> -0.0859
#>  0.0803
#> -0.0397
#> -0.0348
#>  0.0588
#> -0.0002
#>  0.1583
#> -0.0854
#> -0.0585
#>  0.0453
#> -0.0162
#> -0.0794
#>  0.2122
#> -0.1836
#>  0.1866
#>  0.0267
#> -0.0624
#> -0.1462
#> -0.0208
#>  0.1445
#>  0.0148
#>  0.1141
#>  0.0613
#>  0.0974
#>  0.1465
#>  0.1363
#>  0.0464
#>  0.0861
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{100} ]
```
