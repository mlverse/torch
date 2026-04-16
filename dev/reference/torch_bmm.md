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
#> -2.4030  3.6226  2.8038  1.5887 -1.5438
#>  -0.2254 -0.3921  1.1050  0.1357  0.2819
#>  -1.0376 -0.5262  1.2072 -2.6472  1.1184
#> 
#> (2,.,.) = 
#>  0.5515  2.7166  1.5642 -0.4886  1.4492
#>  -0.7184  1.1065  0.5871 -0.8241  0.6489
#>  -1.8186 -2.1025 -4.3229 -0.5302 -1.2458
#> 
#> (3,.,.) = 
#>  1.2536  1.2271 -1.5495  0.6633  0.1950
#>  -1.3215 -1.1021  2.1381 -0.4498  2.0289
#>   0.3356 -2.4539 -3.5515 -0.7670 -2.7083
#> 
#> (4,.,.) = 
#>  0.9929  0.7463  1.5353 -2.5544  2.4770
#>   0.8596 -0.9427  1.3447 -0.5392  1.6718
#>  -1.4571  0.2835  0.7986  0.5247  2.7585
#> 
#> (5,.,.) = 
#>  0.1804  1.0596 -1.0764 -2.2646 -0.5752
#>   0.4499  0.6454 -0.7865 -0.9966 -0.1521
#>  -1.1166  1.6096 -0.6333  0.0867 -2.2905
#> 
#> (6,.,.) = 
#> -3.1393 -2.3607  0.6483  1.9547  0.1428
#>  -3.9916 -5.5693  1.3389  3.3785  0.3182
#>  -2.5105 -1.2742  0.2316  1.8881  0.1901
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
