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
#> -0.2627  1.5113  1.6680 -3.1617 -1.6055
#>  -0.3215  0.1962 -2.2157 -3.3335  0.7198
#>   0.1873  0.2595 -0.4298 -2.8107  0.6941
#> 
#> (2,.,.) = 
#> -1.2307  0.1976 -1.1268 -1.0405 -1.7726
#>   0.0711  0.7471  0.5873  1.4166  0.7233
#>   2.5368 -1.3259  0.3793 -0.2138  0.4018
#> 
#> (3,.,.) = 
#> -1.9640 -0.0289  2.7209  1.4984  1.4456
#>  -0.6909  2.0400  0.0941 -1.3393 -0.5303
#>   0.5158  2.2449  4.3547  4.6025 -0.3149
#> 
#> (4,.,.) = 
#> -0.5751 -1.6307  2.3720  0.1596 -1.3310
#>   1.1266 -0.8270  1.2756  1.4638 -0.0690
#>  -2.4986 -3.5610  1.6418  0.0033 -0.8532
#> 
#> (5,.,.) = 
#>  0.3349  1.6961 -0.0912 -3.1407  0.5679
#>   1.6275 -2.1341 -2.0248 -1.5531 -0.3634
#>   1.0636  0.8162 -1.4331 -0.5020 -1.0943
#> 
#> (6,.,.) = 
#> -1.4653  2.8310 -0.7702 -0.7029 -1.0756
#>  -2.0658 -0.5307  2.4323  0.1166 -0.0541
#>   3.8504  1.1856 -2.4346  1.3700 -1.4658
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
