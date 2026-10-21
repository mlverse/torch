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
#>  2.9876  0.3707  3.0067 -0.9129 -0.3939
#>  -1.1462  0.6383 -6.1434  3.2307  0.1654
#>   1.6674 -1.1806 -2.3735  2.8092 -2.0327
#> 
#> (2,.,.) = 
#> -2.1116  3.7924 -0.2209  1.6179 -0.4950
#>   0.5754 -1.3369  0.3578 -0.7797  0.9703
#>   1.6689  0.7173 -0.0937  0.0549 -1.2488
#> 
#> (3,.,.) = 
#>  0.9968  0.3831 -0.5062 -0.1897  1.0880
#>   3.2550  1.8337 -0.0073 -1.4474 -2.0437
#>  -0.4519  0.1921  0.3983  0.6517 -0.1299
#> 
#> (4,.,.) = 
#> -3.1773 -0.0211 -1.9662 -2.8325  0.3854
#>   0.0490 -0.6948 -0.9400  4.1314 -0.9571
#>  -3.2585  0.1026 -1.1803 -3.3796  0.4966
#> 
#> (5,.,.) = 
#>  0.1485 -1.7066 -0.9493  1.4136 -2.4965
#>   2.5054  0.9868 -0.7253 -0.5192 -4.2332
#>   1.3689  2.0655 -0.8698 -1.9699  2.0737
#> 
#> (6,.,.) = 
#>  0.2553  1.0894  5.3124 -0.3300  0.6970
#>  -2.3387 -0.1274 -2.7667  1.0898  2.3538
#>  -5.4475  0.3034 -6.0367  2.0794  3.1094
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
