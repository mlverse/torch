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
#>  -0.4309  1.8046  2.1233 -0.1220  1.3599
#>   0.2135 -3.1217  0.3793  1.3217 -3.1192
#>   0.0310 -0.7484 -1.9196  0.1282 -1.0006
#> 
#> (2,.,.) = 
#>   2.6845  0.0107  3.7533  2.2380  1.8912
#>  -2.2436  1.5162 -2.7456 -0.6660 -0.7886
#>  -0.9220  1.5797 -2.0457  0.4792 -0.7792
#> 
#> (3,.,.) = 
#>  -0.6084  0.3452 -0.3032  0.9129 -2.5965
#>   0.5102  0.2970 -0.6902  0.9570 -1.4859
#>   0.5002 -0.8959  1.3061 -0.2663 -1.5809
#> 
#> (4,.,.) = 
#>  -3.7294 -0.6207 -0.4546 -1.6836 -1.2503
#>  -3.1521 -0.8873 -0.3645 -1.9482 -0.3700
#>   5.2204  0.2293  2.1503  2.5138  2.0103
#> 
#> (5,.,.) = 
#>   1.1270 -3.3054  0.9374 -2.9005 -0.4782
#>  -4.0744  0.4455 -2.4800  1.0979  1.0186
#>  -1.3283  1.9490 -0.5535 -0.1121  0.3576
#> 
#> (6,.,.) = 
#>   1.0073 -0.1991 -0.2458  0.6973  1.3861
#>   0.8503 -0.6028 -0.0916  1.2005  1.9218
#>   1.6495  1.2925 -0.4686 -0.5307  1.9852
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
