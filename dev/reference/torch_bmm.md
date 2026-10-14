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
#>  0.4607  1.2380 -1.8687  0.1136 -0.3200
#>  -0.9065 -3.0576 -1.5263  0.4221  0.6863
#>  -0.8190 -3.2681  1.2441  1.2806  1.0774
#> 
#> (2,.,.) = 
#> -2.3415  3.6803 -1.6941 -0.6273  1.8143
#>   0.6539 -3.8032  0.2079  1.1732 -1.6772
#>  -3.2230  4.7021 -2.2063 -1.1958  2.2205
#> 
#> (3,.,.) = 
#>  0.6543  0.4602 -4.4099 -2.7841  2.8310
#>  -1.8373 -0.4084  3.3505  1.2422 -1.8304
#>  -1.6451 -1.2876  1.9674  0.6886  1.1641
#> 
#> (4,.,.) = 
#>  1.5941 -0.7158 -2.9913 -3.3635 -1.8890
#>  -2.0347 -2.4026  2.4259  4.3856  3.9319
#>   0.2292 -1.4931 -0.2289 -0.3992  0.1957
#> 
#> (5,.,.) = 
#>  0.0164  1.9278 -0.0124  0.2663  1.9330
#>  -1.1668  3.0892 -0.8903  0.1881  0.4116
#>   1.2664 -1.1621 -0.8088 -0.2853 -0.4915
#> 
#> (6,.,.) = 
#> -3.5838 -1.4454  1.4855 -1.6585 -0.3341
#>  -2.6530 -1.9574  0.3567  0.2112 -0.7376
#>  -0.0506  0.9808  2.3213 -0.4784  1.2346
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
