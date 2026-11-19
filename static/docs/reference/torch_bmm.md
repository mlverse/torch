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
[`torch_matmul`](https://torch.mlverse.org/docs/reference/torch_matmul.md).

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
#> -3.8848  2.7502 -1.2434 -1.3128 -1.6147
#>  -1.5992 -0.3163  0.0323 -0.6888  1.2009
#>   0.1584 -0.2230  0.8739  0.1411 -0.1338
#> 
#> (2,.,.) = 
#> -0.7949 -3.3989  2.4679  2.3033 -2.3908
#>  -1.6021  0.9771  0.7571 -0.4258 -0.4047
#>  -5.3042 -1.6492  3.7443  0.4128 -2.8976
#> 
#> (3,.,.) = 
#>  0.6495  0.2250  4.3231 -2.5974  2.7159
#>  -1.7518  1.7870  1.0224 -0.0279  2.3480
#>   3.1819 -2.2241  1.7755  1.4333  2.2482
#> 
#> (4,.,.) = 
#> -1.9333 -3.6490  1.0926 -2.4294 -1.7860
#>  -2.9119 -3.0739  1.9066 -1.9839 -2.7848
#>  -1.5173 -4.5608  1.2537 -0.3490  0.5921
#> 
#> (5,.,.) = 
#>  2.7817  0.7892  0.2496  1.1775  2.0645
#>   1.2279  0.3515 -0.4237  0.1454  1.9406
#>   1.1510 -2.1295 -2.3852 -1.6945  0.1650
#> 
#> (6,.,.) = 
#> -3.4255  1.9104  3.8860 -2.3026 -1.9182
#>   1.0353  1.1461 -1.8551  1.4676  0.8436
#>  -9.4525  4.0049  4.7237  3.9644 -1.8824
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
