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
#>  3.1489  0.9933 -3.8545 -1.5375  4.0740
#>  -2.2828 -0.2517  4.6297 -2.6699 -2.8489
#>  -1.7808 -0.6740  3.3743  0.6851 -1.5530
#> 
#> (2,.,.) = 
#>  3.8969 -2.1787 -1.3240  0.2996 -3.2381
#>  -0.2272  0.4327 -4.1239  1.5753  1.0360
#>   1.9572 -4.5129  0.0619  0.2307 -7.1684
#> 
#> (3,.,.) = 
#> -2.2082 -0.4679  2.3165 -3.0097 -4.0933
#>   1.2415 -0.4922 -3.0706  1.2228  2.9154
#>  -1.1199  0.0738 -0.0864 -0.8341 -1.2553
#> 
#> (4,.,.) = 
#>  1.7160  1.0362 -0.8424 -1.1184 -0.4963
#>   0.5729 -1.9996 -2.6329  0.4415  0.7637
#>  -0.7095 -1.1652 -0.2308  1.1342  0.8983
#> 
#> (5,.,.) = 
#> -1.7096  0.3472  0.0856 -0.9330 -0.5914
#>   0.7007 -2.6885 -0.8053  0.5924  2.7296
#>  -4.9965  0.8444 -0.0998 -3.0945 -1.1490
#> 
#> (6,.,.) = 
#> -0.4413 -2.8426  1.6928 -3.0165  5.5659
#>   0.2465  0.1237  0.4893  1.4186 -1.0017
#>  -1.8375  2.8286 -1.8997  1.3272 -2.7876
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
