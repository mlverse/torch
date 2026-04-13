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
#> -1.0427 -1.5937  0.9874  4.0675  0.1837
#>   0.5961 -1.0093 -1.8797 -2.1257 -0.7857
#>  -0.2498 -2.7045 -0.7336  1.7358 -2.3461
#> 
#> (2,.,.) = 
#>  0.5456  0.6276  0.3889 -1.2844 -1.2510
#>  -2.7452 -0.2042 -1.2063  4.0709 -0.6254
#>  -0.7152 -0.3812 -0.5645  0.3544  2.2314
#> 
#> (3,.,.) = 
#>  0.9986  0.9570  0.7325 -0.7715  0.2699
#>  -6.0811 -1.3170  0.7632  0.6544 -0.8480
#>   1.9451 -0.8070 -2.2487 -0.7337  0.1855
#> 
#> (4,.,.) = 
#>  0.9056  0.5873 -0.6372 -0.0976  0.2999
#>  -3.3684 -4.0963  1.1392 -5.7131  0.5651
#>   1.6001  0.6626  0.7076 -1.2908  2.0914
#> 
#> (5,.,.) = 
#> -2.8853 -2.2192 -1.0181 -5.2208 -0.2560
#>   6.6361  4.4919  4.5346  6.4440  2.0148
#>   2.8309  1.3709  2.6689  1.8900  1.4606
#> 
#> (6,.,.) = 
#> -2.1136  2.3046  0.0613 -0.2779 -2.8044
#>  -1.9812  2.0190  0.3265 -0.1571  0.4594
#>  -3.7140  5.3767  0.7802  1.4553 -2.1713
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
