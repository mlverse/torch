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
#> -1.4539 -1.2477  0.5300  0.3774  2.1233
#>   0.4310 -0.1184 -1.5280  0.5520 -1.4826
#>   1.8212 -0.0610  4.4853 -1.3365  1.1213
#> 
#> (2,.,.) = 
#> -2.9460 -0.5155  0.9029  1.3269  1.6354
#>   0.1949 -3.4061 -2.1829 -1.8603 -0.2149
#>   3.6375 -5.0382 -3.9989 -5.0755 -1.1079
#> 
#> (3,.,.) = 
#>  2.3879 -1.1382 -3.4622 -1.0717  1.0091
#>  -1.8560  2.7372  0.8753 -2.6158 -0.8177
#>  -0.5035  2.2506 -2.6383 -5.3839 -0.0811
#> 
#> (4,.,.) = 
#> -1.3372  0.6171  0.3972 -0.0206 -1.4472
#>   1.4520  2.3620 -0.9443 -0.3541 -0.3166
#>   2.1037  1.6847 -0.1628 -1.4598  1.3609
#> 
#> (5,.,.) = 
#>  0.8767  1.2909 -0.9980 -0.7029 -1.1251
#>  -1.0889  0.8236 -1.3932  0.9983  1.5991
#>  -0.0109  1.2327  1.5916 -1.5191 -1.4196
#> 
#> (6,.,.) = 
#>  0.2940 -5.9752 -2.1921 -0.6290  1.5205
#>   0.1413 -0.1082  0.5822  0.7655  0.5922
#>   0.0969 -0.8985  1.9755  0.3434 -0.0603
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
