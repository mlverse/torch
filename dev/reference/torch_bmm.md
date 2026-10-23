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
#>  1.1741  0.2559  2.1010  0.6791 -1.0063
#>   0.7230 -0.4138 -1.4034 -2.1578 -0.4895
#>   1.5710 -1.2876 -0.0107  0.0330  5.4184
#> 
#> (2,.,.) = 
#>  0.8388  1.9660 -1.0262 -1.2845  0.1872
#>  -1.8714 -0.5525 -0.2344  1.1003 -0.2241
#>   0.3883 -2.4709  1.6875  0.9729 -0.3365
#> 
#> (3,.,.) = 
#>  1.2832 -1.9016 -0.3001  0.8860  1.2404
#>   1.2865 -1.1586 -2.4160  0.1655  1.4697
#>   0.4389 -0.8549  1.6662  0.6197 -0.9461
#> 
#> (4,.,.) = 
#>  0.9855 -1.6766 -4.0700 -4.0975  4.0677
#>   3.6440 -3.8734 -1.2834 -0.9540  0.8312
#>   1.8192  0.8728  1.5987  3.2078 -0.5403
#> 
#> (5,.,.) = 
#> -4.1807  3.4626  2.6183 -4.4766  1.8168
#>  -0.2935 -1.0232  1.7067  2.0395  0.8639
#>   1.1046 -8.8380  0.8910  0.0105 -0.9801
#> 
#> (6,.,.) = 
#> -1.4214 -0.2557  0.7621 -2.0962  1.5151
#>  -2.0770 -0.0136  0.9170 -0.4942  2.2183
#>  -1.0848 -0.4747 -0.1055  0.8999  1.6079
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
