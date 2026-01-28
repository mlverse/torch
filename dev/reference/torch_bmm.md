# Bmm

Bmm

## Usage

``` r
torch_bmm(self, mat2)
```

## Arguments

- self:

  (Tensor) the first batch of matrices to be multiplied

- mat2:

  (Tensor) the second batch of matrices to be multiplied

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
#>  -2.6310 -3.6420 -0.6343 -0.1225 -0.7739
#>  -2.4083 -1.4316 -0.6167  1.3534 -1.2447
#>  -0.8595 -4.2717  2.7093  2.1820 -1.9595
#> 
#> (2,.,.) = 
#>  -3.9218 -4.6426 -1.4301  3.0448  0.2941
#>  -2.4943 -3.7548 -1.7731  3.0138  1.1571
#>   0.3128  0.4062  0.6809 -0.7936  1.1850
#> 
#> (3,.,.) = 
#>   3.4908 -0.6008  3.4168  0.0779 -0.3182
#>   2.4896 -2.3162  2.1040 -1.9631  1.0977
#>  -0.8057  0.4800 -0.6933  0.3946 -0.1865
#> 
#> (4,.,.) = 
#>   2.3106 -3.2597  0.6912 -1.4014 -3.7356
#>  -2.7772 -1.0003 -1.2223 -2.0665  2.0624
#>  -2.1169 -2.1809 -1.4219 -4.4515 -1.6263
#> 
#> (5,.,.) = 
#>   0.9868  0.2078  0.6681 -0.2274 -0.0728
#>  -0.6201  1.1729 -0.6391 -1.4829 -0.1650
#>  -5.4022 -1.7382 -2.4377  2.1215  0.8480
#> 
#> (6,.,.) = 
#>  -0.7980  0.3044 -0.4854 -0.3638 -0.2244
#>   1.7681  0.0300  1.2374  0.6915  1.0364
#>   1.4697  0.1190  0.4421 -0.1683  0.6256
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
