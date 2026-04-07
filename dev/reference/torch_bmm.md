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
#> -0.3865 -0.0567  0.4351 -1.4702  1.1225
#>   0.1607  0.2424 -0.1637  0.8295 -0.8570
#>   0.4261 -0.0596 -0.6837 -1.6200  1.3640
#> 
#> (2,.,.) = 
#>  2.0728  2.5729  0.2529 -1.6433  0.4885
#>   1.3450 -4.9719 -1.4776  2.9800  1.9885
#>   3.8998 -1.5713 -1.9563  0.5297 -0.9037
#> 
#> (3,.,.) = 
#> -3.7368  0.4434 -0.6793 -4.3418  3.2970
#>   0.7424  0.3534 -1.3977 -5.0264  2.6369
#>  -1.1827  2.0437 -1.7235 -1.1201  0.3814
#> 
#> (4,.,.) = 
#> -0.9368  2.6083 -1.0331  0.1832 -2.4857
#>   0.5172  2.4519 -1.6903  0.4315 -2.0427
#>   0.9673  3.9316 -1.9436 -0.4230 -3.8737
#> 
#> (5,.,.) = 
#>  2.1147 -1.2295  2.9359 -3.0025  1.3430
#>  -0.4228  3.8158 -2.7679  2.0545  0.3341
#>   0.5640  0.6826  0.1862  1.9687 -0.5719
#> 
#> (6,.,.) = 
#>  0.8689  3.1860  2.5462  1.0833  1.5355
#>   2.6535  1.5111  2.4156  1.2830 -0.1828
#>  -2.4898 -0.2528 -3.8926 -2.2295  2.5204
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
