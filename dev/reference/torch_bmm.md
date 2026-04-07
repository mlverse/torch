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
#> -1.8720  0.8362 -0.6766  6.2258 -1.1432
#>   2.0997  4.4625  1.5755 -1.8887  1.2505
#>   1.9567  3.7789  1.2088  2.1522  0.7693
#> 
#> (2,.,.) = 
#>  1.6262  0.8911 -1.5154  1.8149 -0.7774
#>  -2.1496  0.2509 -2.8977 -1.9415 -1.9228
#>   0.2376 -0.8393  1.4972  0.3683  0.7072
#> 
#> (3,.,.) = 
#>  0.8227  1.5578  0.9144 -4.5322  1.1403
#>   0.3689 -1.1483 -2.5736 -0.3493 -1.2284
#>  -1.8372  1.5963  0.6544 -2.5597  0.8820
#> 
#> (4,.,.) = 
#>  0.2627  2.5292 -1.5487  1.6728 -0.5422
#>   0.0910 -0.8841 -3.7929  2.7122 -0.1065
#>  -0.3809 -2.3071 -0.6096  3.7220 -0.7928
#> 
#> (5,.,.) = 
#> -0.0838  2.3906  1.5701  0.5124  0.7357
#>  -2.8168 -0.7711 -2.7159 -0.7000 -0.3241
#>  -0.1356  0.6899  0.1418  0.3400  0.3864
#> 
#> (6,.,.) = 
#>  2.2200  2.3524 -0.7026  0.6391  0.9509
#>  -1.8367  0.3480  3.0196  1.5077 -0.8649
#>  -3.0040  2.5841  0.6390  1.7353 -1.8478
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
