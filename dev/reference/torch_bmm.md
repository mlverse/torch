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
#>  1.5673 -0.8826  0.9049 -0.2761  1.5248
#>  -1.8819 -0.3115  0.2119  1.9709  1.1647
#>  -0.6501 -0.5971 -1.0809  2.0199 -0.7228
#> 
#> (2,.,.) = 
#>  0.3995  1.1470 -0.6309 -0.3553 -0.4168
#>  -2.0927 -0.5378  4.1088 -0.3459 -0.8867
#>  -4.7533  4.4344 -0.0012 -4.9825 -2.2225
#> 
#> (3,.,.) = 
#> -0.5013 -1.2290  0.4336 -2.0295  0.9913
#>   6.1679  3.3401 -0.9081  0.9402 -2.2367
#>  -1.7636 -2.1521 -0.3065 -1.2920  0.4633
#> 
#> (4,.,.) = 
#> -2.5441 -0.3369  0.9971  0.1993  1.3325
#>   0.9177  1.0563  1.8618 -2.9639  3.3004
#>  -1.6072  0.9051  0.8511 -1.5841  1.3430
#> 
#> (5,.,.) = 
#> -0.8336 -2.6981 -0.2521 -0.8272  1.0551
#>  -0.2840  1.5093  0.7835 -1.5914 -5.2853
#>   0.8664 -6.4593  1.5992 -2.4026 -0.6235
#> 
#> (6,.,.) = 
#> -1.5964  3.8955 -0.4989  0.3626  1.1094
#>   1.9399  1.7225 -0.8251  1.2173 -0.0079
#>  -4.8735  0.9551 -0.3422 -0.3873  1.8245
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
