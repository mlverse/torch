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
#> -1.9957  1.6109 -0.6736 -1.2121 -0.5995
#>   4.3390 -4.1848 -5.3822  4.1682 -0.6640
#>   1.1112 -1.7903 -0.9766  1.0884 -0.8467
#> 
#> (2,.,.) = 
#> -2.0287 -3.6015  0.8160 -2.0699 -0.3725
#>  -8.0546 -1.9576 -1.9895  1.6190  1.1521
#>  -1.8747  3.5256 -2.6003  6.7899  2.0840
#> 
#> (3,.,.) = 
#>  4.2763  2.4888 -2.9308  2.0432  0.5696
#>  -2.2000  1.6265 -1.7606  1.2612 -3.2153
#>  -0.8629 -1.2041  1.6103 -1.3675  1.0926
#> 
#> (4,.,.) = 
#>  3.5582 -8.6853 -5.1325  6.2288  3.8362
#>   2.5079 -2.3377 -0.1367  4.8428  2.1660
#>  -0.5745  1.2368 -0.7113 -4.4227 -1.6822
#> 
#> (5,.,.) = 
#>  3.3784 -2.8997 -5.9313  5.2873  1.3944
#>  -0.1640 -0.0292  3.9218  0.4039  1.5776
#>  -2.8682  1.2785 -2.8635 -1.1068 -2.9966
#> 
#> (6,.,.) = 
#>  0.8415 -0.8513  1.4839  1.3824 -0.9280
#>   1.3805  0.4180  2.1591  2.6767 -1.6981
#>  -0.0979 -3.4645  0.4546 -2.4669  0.6101
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
