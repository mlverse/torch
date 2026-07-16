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
#> -2.5762  0.9064  2.6188 -1.2428 -0.6370
#>  -0.0779 -1.6015  0.8754  2.8514 -2.1953
#>   2.8408  0.8272 -0.9720  1.1550 -2.1220
#> 
#> (2,.,.) = 
#> -1.3157  1.6571 -0.9617  0.3398  2.0435
#>   3.7201  4.2012  0.4687  2.1779  0.4565
#>  -3.7667  0.6676 -1.0373 -1.2998  1.5692
#> 
#> (3,.,.) = 
#>  3.0604 -0.4780  1.8066 -1.7644 -0.2946
#>   0.7258 -1.6699 -0.8432 -1.6230  1.0900
#>  -1.1131 -0.5467 -1.0947  0.3282  0.6238
#> 
#> (4,.,.) = 
#>  0.0938  0.1875  2.7153 -0.0415  0.3297
#>  -1.5512  0.4386  2.5510  1.9756 -2.3535
#>   0.5281  0.2999  0.7445 -0.2121 -1.7245
#> 
#> (5,.,.) = 
#> -0.1095 -0.3222 -1.1873 -0.9615 -7.1227
#>   0.2925  0.9943 -1.5440  0.7750  2.1509
#>   3.9663 -2.1539  3.0488  0.3026 -4.1096
#> 
#> (6,.,.) = 
#> -1.1583  0.4864 -2.9403  2.5869 -2.0625
#>   3.9260  1.9082  1.6521 -0.3094  0.1916
#>   3.7880  1.6600 -0.0927  0.1498  0.1911
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
