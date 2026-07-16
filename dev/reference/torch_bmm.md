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
#>  -1.6296  -0.0575  -1.4459   1.1006   1.3105
#>    2.3704   1.5404  -4.2862   0.7485  -0.0370
#>    4.7998   1.8232 -11.0857   7.0680  -3.3684
#> 
#> (2,.,.) = 
#> -0.9496  0.0198 -0.5453 -1.0818 -0.1110
#>   1.3218 -0.0581 -1.0677  0.1871 -0.4000
#>   3.5521 -2.6355 -1.1558  3.9537 -2.6044
#> 
#> (3,.,.) = 
#>  5.4213  2.4968  2.9904 -0.4165 -2.3405
#>  -1.1736 -0.0276 -0.6144  3.3224 -0.2905
#>  -3.3196 -3.7420 -1.9456 -0.7714  4.6364
#> 
#> (4,.,.) = 
#>  0.0906 -0.2777  0.4022 -0.2822 -0.2201
#>  -3.9207 -0.5101 -0.5158 -0.9230  1.1566
#>   1.2313 -0.5039  0.9200 -0.6778 -0.4079
#> 
#> (5,.,.) = 
#>  1.4930  3.0706 -0.6859 -0.0037 -2.0796
#>  -1.2064 -1.3990 -4.4661  2.7180  2.4413
#>  -1.7876  0.0839  0.9547  0.0766 -1.0242
#> 
#> (6,.,.) = 
#> -2.0799 -1.3731  0.7544 -8.4586  0.7986
#>  -0.4567  0.5513 -2.7371  4.0686 -1.3286
#>   0.4289  1.2011  0.2108  0.3033 -2.0034
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
