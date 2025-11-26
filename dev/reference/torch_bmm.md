# Bmm

Bmm

## Usage

``` r
torch_bmm(self, mat2)
```

## Arguments

- self:

  (Tensor) the first batch of matrices to be multiplied

- mat2:

  (Tensor) the second batch of matrices to be multiplied

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
#>  -3.6267 -3.4248 -1.1296  1.4619 -0.7958
#>  -1.0174  0.5212  0.7518  1.2633  2.4241
#>  -1.6566 -0.6057  3.0567  0.3778  4.5200
#> 
#> (2,.,.) = 
#>   0.5667  1.9291 -1.2361  0.4843  1.1561
#>  -0.2102 -0.6257  0.2405 -2.9351 -0.1445
#>   0.3360  1.4532  1.0659 -0.0018  0.3969
#> 
#> (3,.,.) = 
#>  -0.2613 -0.2286  0.2465 -0.4254  0.3944
#>   3.3420  0.8075  1.6443  0.4828  2.4843
#>  -1.6680  1.2368 -0.8161 -1.0021 -0.6819
#> 
#> (4,.,.) = 
#>  -0.6463  2.6146  2.0112  1.5296  0.8311
#>  -0.2955 -0.8299 -0.8987 -0.3475 -0.3428
#>   0.8056  2.8988  3.4594  1.0303  1.5502
#> 
#> (5,.,.) = 
#>  -2.6763 -0.1593 -0.5094 -0.7040 -1.6148
#>  -3.5710  1.3256  2.5879 -0.5335  1.3598
#>   1.0141  0.4768  2.0305 -2.9157 -7.9231
#> 
#> (6,.,.) = 
#>  -0.1238  0.1229  0.4080  1.2976 -1.1609
#>   0.4686 -0.5675  0.2031  1.3019 -0.3635
#>  -0.8633  1.7421 -1.8361  1.2408 -0.1111
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
