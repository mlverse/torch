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
#>  3.6833 -5.2152 -0.8454 -1.3605 -1.9634
#>   3.3553 -2.1643 -1.7197 -1.4829 -1.5444
#>  -0.2498 -1.4017 -0.3934  0.5585 -0.3228
#> 
#> (2,.,.) = 
#> -1.5662 -3.5007  1.2738 -3.4661  2.3585
#>  -4.1072 -2.3645 -3.1860  2.6176  4.5766
#>   0.5833  1.6354 -2.3334  3.0939 -0.8580
#> 
#> (3,.,.) = 
#> -1.5506 -0.2589 -1.2048 -0.1504  0.2481
#>   0.5314  0.1442 -1.0988  0.6903 -0.1703
#>  -1.2256  0.1450 -2.6029  0.0503  1.3920
#> 
#> (4,.,.) = 
#> -3.1213 -1.0551  4.4509  2.3871 -0.7070
#>   0.6147 -0.4909 -0.3088  0.5344 -0.4888
#>  -1.3160 -1.3825  2.3964 -1.3086  1.6890
#> 
#> (5,.,.) = 
#> -2.9402  0.3767 -0.9877  0.5790 -2.3713
#>   3.4792  0.5279  0.6127 -0.3792  0.4439
#>  -4.2370 -3.2017 -0.7294  1.4907 -1.2198
#> 
#> (6,.,.) = 
#>  0.3395 -0.0218  0.4455  2.0816  0.5751
#>   0.0333  1.4203  0.1342 -4.8904 -2.2458
#>  -0.4934 -1.6112  0.2929  6.3719  2.4069
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
