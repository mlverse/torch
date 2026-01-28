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
#>  -3.7956 -0.3880  1.8628  0.3257 -0.5043
#>   0.9021 -0.2961 -1.1154  0.1427  0.2929
#>   0.2171 -2.2364 -1.3826 -1.6658 -2.3839
#> 
#> (2,.,.) = 
#>   2.4354 -1.2175 -2.1099 -0.5585 -0.2739
#>   1.6298 -0.0616  1.7920  0.4230  1.0291
#>  -2.1120 -1.4804  1.5193  0.0695 -0.0051
#> 
#> (3,.,.) = 
#>   1.5011 -1.4219  1.4986  2.5272 -2.5623
#>   2.4433 -0.6391  1.5977  0.8338 -1.2042
#>  -1.8748  0.3829 -1.4674 -0.0191 -0.4845
#> 
#> (4,.,.) = 
#>   1.8111  0.0434  1.4134  0.3140  1.3316
#>  -1.1864  0.5715  0.9173  0.7366  0.7517
#>   0.1907  0.6917 -0.1667  0.1623 -0.0664
#> 
#> (5,.,.) = 
#>  -1.7619  1.1469  0.5653 -0.7096 -3.5012
#>  -0.8129  0.0707 -0.8648 -0.3970 -0.3743
#>  -0.3791  0.8726 -2.8612  0.8808  2.5189
#> 
#> (6,.,.) = 
#>  -0.1653 -1.4550  0.2450 -1.6234  0.3392
#>   2.1693  5.1246  1.4204  8.1488 -2.1066
#>   0.7920 -1.2479  0.0651 -2.1368  2.9883
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
