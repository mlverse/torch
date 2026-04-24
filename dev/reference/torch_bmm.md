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
#>  0.3238  0.0159 -0.2847 -0.7121 -1.7302
#>  -0.6430  1.6063 -1.8632  2.9237  0.8416
#>   0.5071 -0.3812 -0.3232  0.7507 -2.1728
#> 
#> (2,.,.) = 
#>  0.6350 -0.5117 -0.5344 -0.8070  2.3044
#>  -1.1079  0.2983  0.8216  0.8045 -2.8381
#>   1.7535  0.7286  1.8228  0.9678 -0.9486
#> 
#> (3,.,.) = 
#> -0.2217  1.3179 -1.0186  4.2635 -2.1222
#>  -0.8119 -0.1766  0.7358 -4.7593  1.9738
#>   0.9292  0.3175 -1.1128  1.6289  0.3335
#> 
#> (4,.,.) = 
#>  2.4306  5.6385  0.4278  5.7238 -0.0027
#>   0.3114 -0.6050  0.5159  0.5515 -0.7611
#>  -1.3778  1.6743  1.0034 -0.0966 -1.7559
#> 
#> (5,.,.) = 
#> -2.4393  0.3968  0.8227 -1.4462 -1.6602
#>  -3.6852  1.2182 -2.1198 -1.3466  0.5353
#>  -2.5944  0.6816 -0.8751 -1.1056 -0.2033
#> 
#> (6,.,.) = 
#> -1.3860 -0.2991  0.0722  0.1214  0.2296
#>   2.1883 -1.6161  0.0462 -0.5623 -1.4450
#>   0.6713  0.2152 -0.4153 -0.0897  0.1287
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
