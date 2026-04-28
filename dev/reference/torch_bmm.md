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
#>  1.9130  0.6033  0.3888  0.5372 -0.4651
#>  -3.1482  1.9479 -1.2457  1.1683  2.4823
#>  -0.1129  0.9505  0.3201 -2.4624 -2.6049
#> 
#> (2,.,.) = 
#>  0.8762 -2.0045  1.3103 -0.4235  1.7788
#>   3.9461  4.8694  3.7881  1.6393 -1.5353
#>   0.6971  0.9571  0.2847 -0.7819  0.1743
#> 
#> (3,.,.) = 
#> -1.5990  2.3937  1.9001  0.6660  4.9165
#>   1.9670 -1.1894  0.9631  1.7993  0.8822
#>   0.6011 -1.0120  0.2467 -1.6122 -3.6152
#> 
#> (4,.,.) = 
#>  0.5338 -3.1084 -0.5995  1.0252  1.5329
#>  -1.7698  3.4609  0.7087 -1.8167 -3.1960
#>  -0.2377 -1.5909 -0.1485  0.1318  0.2653
#> 
#> (5,.,.) = 
#>  0.1142  1.2295 -1.1071 -0.0130 -0.5943
#>  -0.3616  2.6841 -0.9658  0.5320 -1.3257
#>   0.8364  0.6762  1.8894  1.4191  2.2782
#> 
#> (6,.,.) = 
#> -3.0720 -0.5186 -0.3124  1.1297 -4.3022
#>   2.3711 -0.7139 -0.3969 -2.4186  1.5882
#>  -0.4344 -1.0548  0.2255  0.0597 -0.6742
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
