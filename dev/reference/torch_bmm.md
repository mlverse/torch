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
#>  1.6151  0.1707  1.7293  0.6995  1.6525
#>   1.2082 -2.1328  4.7946  2.2495  2.0327
#>   1.7649  1.0495  2.5914  1.9857 -3.7194
#> 
#> (2,.,.) = 
#>  0.6949 -1.4974  0.3950  1.0431 -0.3912
#>   0.2003 -1.6285 -6.8481  0.5077 -2.8479
#>  -0.7869 -0.2342  8.3659  1.0105  3.6231
#> 
#> (3,.,.) = 
#> -2.8190 -1.1422  0.5051 -0.0619 -0.0932
#>   1.4640  0.7552  0.4303 -0.9524  0.2426
#>   5.3465  2.3318  1.0088 -0.6919  0.0781
#> 
#> (4,.,.) = 
#> -1.1569  0.4262  0.7931 -0.8389  2.1221
#>  -0.1769 -2.7561  0.8206  0.2959 -0.6604
#>   0.6364 -0.2305  4.3691 -3.0509  0.8427
#> 
#> (5,.,.) = 
#> -1.9145  0.2741 -2.4605 -1.5981  0.4895
#>  -1.6252 -0.5638  0.7977  1.7174 -0.8872
#>   3.0256  0.2319  3.6085 -1.7763  2.1634
#> 
#> (6,.,.) = 
#> -1.6340 -5.7147  1.2653 -1.5143  0.9071
#>  -1.5453 -4.9357 -2.5827  2.0786  4.8624
#>  -0.6772 -1.8800 -1.0533  1.0010  1.9020
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
