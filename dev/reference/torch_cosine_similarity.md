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
#> -0.1013
#> -0.0593
#>  0.1635
#>  0.1809
#>  0.0548
#> -0.0620
#>  0.0409
#> -0.2270
#>  0.0078
#> -0.0431
#> -0.0268
#> -0.1058
#> -0.0608
#> -0.0845
#>  0.1579
#>  0.0602
#> -0.0109
#> -0.0382
#> -0.0468
#>  0.0651
#> -0.0300
#>  0.0670
#> -0.0151
#>  0.0894
#>  0.0374
#> -0.0951
#>  0.0015
#> -0.0679
#> -0.0801
#> -0.1150
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{100} ]
```
