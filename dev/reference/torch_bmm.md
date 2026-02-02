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
#>  -0.2250  0.0495  0.6492  1.1993  0.6343
#>   2.5574  0.7145 -2.8556 -3.5121 -0.3296
#>   0.3650 -2.7745  0.0454 -2.4041  0.0965
#> 
#> (2,.,.) = 
#>   2.1910  0.1012  1.9856  0.7720  3.4013
#>   0.1450  0.6831 -0.4878  0.3450  1.2931
#>   1.4669  0.6619 -2.3020  2.0332 -0.0419
#> 
#> (3,.,.) = 
#>   0.1764 -0.1416 -1.1952 -0.1719 -1.2887
#>   1.5423  3.7994  1.4440 -1.0765  0.9532
#>  -1.4023  2.0034 -0.6780 -2.4120 -0.4183
#> 
#> (4,.,.) = 
#>   6.2365 -1.2271 -1.9834 -1.4609 -0.7383
#>  -0.0225 -1.8221 -0.5059 -0.6008 -0.8111
#>  -0.0198 -0.8889 -1.1788 -1.4724 -0.2280
#> 
#> (5,.,.) = 
#>   0.1161 -1.7801 -4.1373 -0.4132 -1.1811
#>   1.3487  0.4934  0.6036 -0.3456  1.9862
#>  -1.0002 -0.3721 -1.6104  0.1670 -1.9376
#> 
#> (6,.,.) = 
#>  -1.7108  2.7055 -5.3887 -1.8485  1.7319
#>  -0.8790  2.1241  0.2681  0.1192 -0.6262
#>  -0.0491  1.5053 -1.3052 -0.5340  0.7709
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
