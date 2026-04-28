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
#> -2.0681  1.7430 -4.0659  0.3304 -0.1585
#>   1.8814 -1.7919  2.8937 -0.6479 -0.8995
#>  -1.5325  0.7514 -3.9148  1.4465 -0.2129
#> 
#> (2,.,.) = 
#>  0.6977  0.3636  1.2429  1.4354 -0.6553
#>   0.2843 -1.1496 -2.0023  0.1087 -1.4829
#>  -0.2332 -0.0620  0.6840 -0.2766  0.9571
#> 
#> (3,.,.) = 
#>  0.9119  2.8121  2.2784 -0.4947 -1.7577
#>  -1.9016 -1.4356 -1.1053  0.4050  1.7705
#>   1.1815 -1.7942 -0.1183  2.7826  1.3640
#> 
#> (4,.,.) = 
#>  1.0641 -2.7590  1.4036  0.3885  0.2035
#>  -2.8188 -0.7215 -1.5892 -1.4837 -0.7461
#>   3.8869  4.6457  2.2083 -0.0065  0.4804
#> 
#> (5,.,.) = 
#> -2.2383 -0.2461  0.7751  0.1987 -0.4658
#>  -1.5436 -1.9285 -2.1467  1.0036 -0.2993
#>   1.7715  0.6218  1.3506  2.6616  2.2787
#> 
#> (6,.,.) = 
#> -2.4305  3.8337 -0.7297  0.0248 -0.5350
#>  -0.5632  2.4869 -0.3816  1.2869 -2.1826
#>  -0.1363  0.9363 -0.1685  0.6174 -0.9262
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
