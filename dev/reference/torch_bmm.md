# Bmm

Bmm

## Usage

``` r
torch_bmm(self, mat2, out_dtype)
```

## Arguments

- self:

  (Tensor) the first batch of matrices to be multiplied

- mat2:

  (Tensor) the second batch of matrices to be multiplied

- out_dtype:

  (torch_dtype, optional) the output dtype

## Note

This function does not broadcast . For broadcasting matrix products, see
[`torch_matmul`](https://torch.mlverse.org/docs/dev/reference/torch_matmul.md).

## bmm(input, mat2, out=NULL) -\> Tensor

Performs a batch matrix-matrix product of matrices stored in `input` and
`mat2`.

`input` and `mat2` must be 3-D tensors each containing the same number
of matrices.

If `input` is a \\(b \times n \times m)\\ tensor, `mat2` is a \\(b
\times m \times p)\\ tensor, `out` will be a \\(b \times n \times p)\\
tensor.

\$\$ \mbox{out}\_i = \mbox{input}\_i \mathbin{@} \mbox{mat2}\_i \$\$

## Examples

``` r
if (torch_is_installed()) {

input = torch_randn(c(10, 3, 4))
mat2 = torch_randn(c(10, 4, 5))
res = torch_bmm(input, mat2)
res
}
#> torch_tensor
#> (1,.,.) = 
#> -0.9716 -0.1924  1.8493  2.3199 -1.3434
#>   0.3964  1.4993  2.9090  3.0751 -1.0022
#>   0.5244 -3.0157 -5.5361 -5.8848 -1.3076
#> 
#> (2,.,.) = 
#>  1.4336  0.1833 -0.6355 -0.4632 -0.1684
#>   1.0171 -0.2021 -0.2617 -0.3826 -0.2952
#>   2.0622  1.0383  1.6271  0.7036  0.7387
#> 
#> (3,.,.) = 
#> -3.1892 -2.2171 -1.7813  1.6084  1.2242
#>  -2.6135  0.4134 -0.6667  0.6414  0.8334
#>  -1.8962 -2.2327 -0.2220 -0.2470  2.2407
#> 
#> (4,.,.) = 
#> -0.2028  1.0008  0.0105 -0.2462 -0.3072
#>  -2.5752 -1.8973 -0.2549  0.3831  0.9236
#>   0.4473 -4.1999  0.1922  1.1513  1.8584
#> 
#> (5,.,.) = 
#>  0.2403  0.5108  0.4383 -0.0322  1.3479
#>   1.3846  0.1563  6.3007  1.3404  1.3028
#>   1.4381  1.0960 -0.9710  0.8047 -0.4681
#> 
#> (6,.,.) = 
#>  2.6243 -0.1896 -1.9488  1.2370  0.4823
#>  -6.1393 -1.1861  2.0898 -7.8600  0.1304
#>   2.4980  0.6498  0.8340  5.5618 -0.0122
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
