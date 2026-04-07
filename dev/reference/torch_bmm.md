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
#> -2.8389  0.1600 -0.4510 -0.3761  1.3740
#>  -0.1454  0.3127  2.5396  0.9006 -0.7468
#>   0.1149  0.4607  0.4035 -0.2026 -1.7242
#> 
#> (2,.,.) = 
#>  2.7824 -1.2399 -0.3860  0.0072  1.6588
#>  -1.3829  0.0249  0.0509 -1.6054 -2.9395
#>   1.3574 -1.4995 -0.5181  0.4473  1.1527
#> 
#> (3,.,.) = 
#>  3.1447  1.0216 -0.0285 -1.4191  0.1025
#>   0.4196  1.6048 -0.4798 -0.0188 -0.9619
#>  -2.0099  3.9146  1.9277  2.7492  0.6181
#> 
#> (4,.,.) = 
#> -1.1311 -2.3917 -0.8620  4.4971  2.1217
#>  -0.6818  2.9362 -4.6149 -0.8362  2.1111
#>   0.0206  0.2861 -0.2916 -0.4324  0.0168
#> 
#> (5,.,.) = 
#> -0.3278  1.3263  0.6441  0.2553  0.7303
#>  -0.5870  2.4004  0.1764  0.3034  1.4576
#>  -3.1765  4.2691  2.4418 -1.3221  2.5644
#> 
#> (6,.,.) = 
#>  0.5029 -0.0543 -0.2531  0.1856 -0.0714
#>  -0.3574 -0.1351  0.5636 -0.9490 -1.7098
#>   0.9731 -0.1900  0.7818  0.4662 -1.5306
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
