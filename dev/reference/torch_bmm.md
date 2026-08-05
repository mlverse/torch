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
#> -1.7854  0.9168 -4.6550 -0.2596 -1.4870
#>  -1.5340  3.1973  6.1263  2.9655  1.6682
#>   0.0337  0.8093 -0.4396  0.3096  0.3819
#> 
#> (2,.,.) = 
#> -1.0825  0.6595  0.7387  1.2950  3.7417
#>   6.2614 -0.0962 -0.5524  4.5329 -1.5591
#>  -1.1537 -0.5606 -2.6763  0.9057  0.9063
#> 
#> (3,.,.) = 
#> -2.5122 -2.1547  5.0504  7.8965 -1.2841
#>   0.4924  0.0128 -1.3988 -0.9284  0.9620
#>  -0.6535 -1.2401  0.7346  2.7456  0.5844
#> 
#> (4,.,.) = 
#>  0.3924 -4.8109  0.2140  1.5382  2.4855
#>   0.4832  3.0139 -0.2460 -2.2837  1.4454
#>  -0.4810  7.1412 -2.2500 -6.6787  3.7346
#> 
#> (5,.,.) = 
#>  1.2442 -3.9705 -1.0057  3.4580 -0.1323
#>   1.2035 -1.0342  2.9925 -1.1613 -2.8276
#>   2.4111 -2.7333  4.2189 -0.5301 -4.0071
#> 
#> (6,.,.) = 
#> -0.5951 -2.5429  2.1712  2.8398 -1.1116
#>  -1.5424  2.9334  0.9482 -2.2570 -3.6654
#>  -0.2502 -0.1423  1.0335 -4.3904 -1.2070
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
