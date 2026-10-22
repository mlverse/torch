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
#>  0.4117 -1.8265 -0.5628  1.0447  0.3214
#>   2.6509 -4.0028 -1.5242  0.8718 -1.7041
#>  -3.3309  3.0335  1.3933 -0.2416  1.5442
#> 
#> (2,.,.) = 
#> -2.1055 -0.1160  0.0175 -1.2201  1.9541
#>   1.0167  0.9840  1.7368  2.1461  3.9037
#>  -0.4470  0.2820 -1.6429 -3.3050 -2.9859
#> 
#> (3,.,.) = 
#>  0.8842  0.3389  0.3883 -1.4662 -2.1833
#>   0.3636  0.0704  1.1139  1.3634 -1.0374
#>  -1.0290 -2.0973 -1.4291  2.4848  8.0125
#> 
#> (4,.,.) = 
#> -0.7942 -1.2141  1.3016 -1.4607  2.5064
#>   0.3480 -0.7737  1.0208 -0.0474 -1.5651
#>  -0.3673 -2.3490  1.4573  0.5547 -2.8074
#> 
#> (5,.,.) = 
#> -0.8143  1.8642  1.1514 -2.2270  2.8218
#>  -0.4139 -0.6051 -1.9851 -0.8673 -0.3987
#>  -0.1683  0.5156  0.4896 -0.6228  0.7955
#> 
#> (6,.,.) = 
#> -3.2745  1.8370 -3.5781  0.8475 -2.3868
#>   1.2267 -0.0995  1.4117 -0.7167  3.9815
#>   1.2421 -0.6046 -1.3319  1.4271  0.4207
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
