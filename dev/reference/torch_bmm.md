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
#> -2.3415 -2.2877 -3.2412 -0.9828 -0.9150
#>   2.9182  3.6392  3.9767  0.7684  0.0489
#>  -0.7309  1.2998  1.9022 -2.4141 -0.3466
#> 
#> (2,.,.) = 
#> -2.5010 -1.3356  0.0204 -0.6766 -0.1008
#>  -1.5747 -1.2277 -1.4982 -0.1736 -1.6653
#>   4.6212  1.3418  2.8047  0.5524  1.6730
#> 
#> (3,.,.) = 
#> -0.3524 -0.4557 -0.3960 -0.0793 -0.4553
#>  -7.5768 -1.8383 -2.1286 -0.4377 -0.0319
#>   2.5435  0.7945 -2.4024 -3.4934 -1.4949
#> 
#> (4,.,.) = 
#>  1.1565 -0.6736 -0.9027  2.2358  0.9627
#>  -0.0313  3.1044  0.5406 -1.8061  0.0519
#>   0.2315  2.5914  1.9448 -0.5407  0.8022
#> 
#> (5,.,.) = 
#>  1.4924  1.2872  0.3199  0.6246  0.3479
#>   1.4008 -0.5110  1.7750  3.6979 -0.2679
#>   0.1235 -2.9581  2.9739  5.3270 -0.5102
#> 
#> (6,.,.) = 
#>  2.2878  0.8574 -0.1931  1.9247  0.5716
#>  -0.4708  0.4246 -0.1103 -0.9990 -0.0263
#>  -1.7493  0.0949 -0.4204 -2.5027 -2.3314
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
