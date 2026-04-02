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
#>   0.5533 -0.7881 -1.7594  0.3312  1.8261
#>  -0.1664 -1.0872  0.4843  3.6363  2.7877
#>   0.4980  0.2066 -0.3867  0.2065 -0.2240
#> 
#> (2,.,.) = 
#>   0.2957  0.2882 -1.0781  0.5638 -0.7390
#>   0.2572  3.4162  0.1964  0.2942 -1.4142
#>  -0.2462  3.5018 -0.7978 -0.1543 -0.8782
#> 
#> (3,.,.) = 
#>   1.5993  0.2962  0.9771  2.3433  1.0476
#>  -1.3595 -2.9449 -2.7730 -2.8434  1.8307
#>  -1.1771 -1.0218 -1.0773 -1.8575 -0.0457
#> 
#> (4,.,.) = 
#>  -2.3139  3.1064 -0.0695  1.2882 -1.0253
#>   0.9280  1.8081 -2.9583 -6.4855  2.6100
#>   2.8966 -2.9055  0.4447 -2.2165  0.6023
#> 
#> (5,.,.) = 
#>   0.3433 -0.7084 -0.1233 -0.4912 -0.2817
#>   1.5707 -4.0770  1.6058  3.6254 -0.4737
#>  -0.8031 -1.7519 -0.4856  0.7470 -0.8591
#> 
#> (6,.,.) = 
#>   2.5438 -5.5064  2.1731  1.6255 -1.5827
#>   0.7965 -4.5996  1.4886 -0.3812  1.0875
#>   0.5910  0.0060 -0.1568  0.7214  0.8710
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
