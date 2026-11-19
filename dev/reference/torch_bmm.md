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
#> -1.4435 -0.1659  1.4509 -8.0886 -0.6522
#>   0.0603  1.7733 -0.5368  1.5232 -0.7151
#>   0.8341  2.2045 -0.2329  3.1984  0.6553
#> 
#> (2,.,.) = 
#> -0.7593 -1.3226 -1.8743 -1.2676 -0.1703
#>  -1.7888 -1.7978 -2.6287 -1.7247 -0.5732
#>  -0.8269  1.4984 -3.4223 -1.1487 -0.6855
#> 
#> (3,.,.) = 
#> -0.3177  2.3626  1.0933  1.8880  1.4915
#>  -1.8855 -0.4355  0.4547  0.9285 -0.5428
#>   0.7111  1.9993 -2.7530  0.2618 -1.9991
#> 
#> (4,.,.) = 
#>  2.6055  1.3781  1.1700 -0.1342  0.6863
#>  -2.5867  0.5064  2.5666 -3.0976 -1.3339
#>  -0.4969 -1.6411 -1.7861 -0.5863 -1.1939
#> 
#> (5,.,.) = 
#>  5.2680 -1.6599 -4.1889 -0.9678 -1.5070
#>  -2.9497  0.4628  2.2210  1.5538  1.7748
#>  -3.9440  2.2721  2.0312  0.1303  3.6194
#> 
#> (6,.,.) = 
#>  0.9118  1.3859 -1.5990  0.5531 -0.9386
#>   1.7206  0.6072 -0.2900 -0.3964  0.3744
#>  -0.1083  0.3964 -1.6054  1.9374 -1.7409
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
