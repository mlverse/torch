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
#>  0.0194
#> -0.0793
#>  0.0188
#> -0.0697
#>  0.1882
#>  0.1536
#>  0.0495
#> -0.0004
#> -0.1532
#>  0.0318
#>  0.0647
#>  0.0943
#> -0.1328
#> -0.0036
#> -0.0019
#> -0.0300
#>  0.0048
#> -0.0503
#> -0.1995
#> -0.0721
#> -0.0867
#> -0.0055
#> -0.0428
#>  0.0376
#> -0.0543
#> -0.0464
#> -0.0279
#> -0.1866
#> -0.0377
#> -0.0195
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{100} ]
```
