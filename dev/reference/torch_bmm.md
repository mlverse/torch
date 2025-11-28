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
#>  -0.7917 -0.5218  0.5310  0.6058  0.1789
#>  -1.4017 -1.1435  0.5094 -0.3153 -0.8175
#>  -0.2818 -0.1170  0.6414  1.3756  1.0188
#> 
#> (2,.,.) = 
#>  -0.6992 -1.0844  3.8239  1.3386 -2.0200
#>   1.2640 -2.5001  1.3144  0.3126  0.5532
#>   0.1304  0.4549 -1.5791 -1.0853  1.2100
#> 
#> (3,.,.) = 
#>   1.1626  1.0954 -0.2497 -0.3179 -1.0635
#>  -0.2488  1.2364  0.7068 -0.6073 -0.1650
#>   3.1571 -0.0609  1.1474 -2.5003  1.2496
#> 
#> (4,.,.) = 
#>   1.3278 -1.2167 -1.0544  0.6302 -0.6340
#>   2.1084  0.4677 -1.3512 -0.1776  1.1192
#>  -1.2447 -4.1967  0.8050  2.9205 -4.2141
#> 
#> (5,.,.) = 
#>   1.8087 -0.3677 -1.2003 -0.5124  1.7364
#>   0.9385  1.9184  2.2859 -2.3012  0.7570
#>   1.1821 -0.5709 -1.8163  0.2442  0.9670
#> 
#> (6,.,.) = 
#>  -0.1648  0.6251  1.4797 -2.5349  0.9006
#>  -0.0610  0.0043 -0.0269  0.1259  0.2373
#>  -1.5545 -1.3412 -1.6137 -1.1040 -1.9997
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
