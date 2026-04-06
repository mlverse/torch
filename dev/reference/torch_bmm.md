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
#>  0.5254  0.3277 -0.7791 -1.7219  0.3357
#>  -4.6938  0.4962  0.0761  0.1069 -0.2723
#>   5.7695 -0.1265  0.7965 -1.2756 -1.4036
#> 
#> (2,.,.) = 
#>  0.3842  0.2793  0.4638  2.3798  1.5001
#>   1.6374 -1.7060 -1.1514  1.2734  1.5141
#>  -3.1155  1.3452  2.6358  0.7091  1.5543
#> 
#> (3,.,.) = 
#> -6.9607  3.1479  6.4122 -2.0994  2.7092
#>   0.8462  3.1433  1.9104  1.3369  1.9300
#>   4.7146 -1.5722 -2.5929 -1.1098 -0.2383
#> 
#> (4,.,.) = 
#> -2.8317 -1.0364  0.8418 -1.9021  1.4157
#>   0.6268  0.0257  2.3597 -1.9672 -2.6904
#>   0.2799  0.2242 -2.5103  2.2348  2.5124
#> 
#> (5,.,.) = 
#> -0.8762  6.6077 -1.8324 -2.3435 -2.1722
#>  -2.1502 -0.3940  1.6649 -2.3014  0.8847
#>  -3.7716 -1.4567 -0.6035  0.4506 -2.2786
#> 
#> (6,.,.) = 
#> -4.6293  2.7443  1.3989  0.4362 -0.0506
#>  -1.5950  4.3370 -1.5436 -1.6221 -0.3751
#>  -0.5856  0.7944 -0.6427 -0.1209  0.3295
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
