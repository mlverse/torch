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
#> -0.0444 -1.4928  1.3880  0.7053  1.2153
#>   5.0202  5.4581 -0.8787 -0.8382 -0.0242
#>  -0.4124  2.5353 -0.9489  0.1840  3.4072
#> 
#> (2,.,.) = 
#>  0.9150 -0.7872  1.1760 -0.5943 -0.4786
#>  -2.3331 -0.6442  1.6144  0.4493  0.8185
#>  -0.3765  0.4279 -0.4356  0.1775 -0.2818
#> 
#> (3,.,.) = 
#> -0.5324 -0.7652  1.3535  1.9907 -0.3233
#>  -0.5352  1.8724 -2.9406 -5.5274  0.1813
#>   2.6564 -3.5933  3.3507  0.7083 -1.5420
#> 
#> (4,.,.) = 
#>  2.1067  1.1194 -1.5029 -1.7578  3.5859
#>   0.1705 -3.8746  0.8068  0.4535  2.3008
#>   0.9930  5.0121  0.0754 -1.6160 -0.8579
#> 
#> (5,.,.) = 
#>  0.2042 -3.6136 -1.2413  1.1775 -1.0234
#>   1.5621 -3.2727 -3.1075 -0.0777  1.5068
#>   2.3691 -4.3110 -5.1297 -0.6675  2.1789
#> 
#> (6,.,.) = 
#>  5.1028 -2.4078 -7.8755  4.6769  0.5214
#>   6.2916 -2.5881 -0.5038  1.9093  3.1440
#>   8.2270 -2.4475 -6.0615  2.4214 -2.4638
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
