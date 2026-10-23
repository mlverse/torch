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
#>  1.8149  0.7601  0.9297 -0.3406  1.3319
#>  -3.8965 -0.2639 -3.4697  0.5616  1.5629
#>   4.8259  0.3710  4.3812 -0.0034 -2.8941
#> 
#> (2,.,.) = 
#>  1.6259  2.4137 -2.6853  0.2204  3.0444
#>  -2.1892 -1.6802 -1.8463  1.7940  0.5922
#>   1.4017  1.0072  1.8727 -1.1122 -0.8462
#> 
#> (3,.,.) = 
#> -3.2460 -1.4621  4.3264 -0.3601  2.0066
#>  -0.9057 -0.9209  2.2011 -0.4144  1.3774
#>   2.6600  0.9679 -4.6055  0.8363 -2.0215
#> 
#> (4,.,.) = 
#> -1.2812 -0.8145  0.6794  0.5913  4.5538
#>   1.4871 -3.8652  4.0862  3.5090  4.9597
#>   0.8942 -1.5844  2.0057  1.2383  1.3782
#> 
#> (5,.,.) = 
#>  0.9867  0.7787  0.7791  0.8849 -0.5302
#>  -2.4990 -0.3472 -0.1047 -0.6522  1.5340
#>   0.4009 -2.7895  0.1409  0.3906  0.0513
#> 
#> (6,.,.) = 
#> -1.3536 -1.8863 -0.3266 -0.1076  0.0391
#>  -3.2840 -3.8242 -2.4676 -1.2795 -0.9836
#>  -1.8617 -3.3293 -0.1461 -0.6265  0.9166
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
