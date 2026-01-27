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
#>   0.5334  0.1676 -3.0083  2.1687 -0.4910
#>   0.1006  0.0840 -2.4851  0.4384  0.1382
#>   0.8882  1.4345 -1.9844  2.5003 -0.7166
#> 
#> (2,.,.) = 
#>  -0.0251  1.0753  0.0250 -0.8175 -0.8461
#>  -0.5346  2.4895 -1.6195 -2.2877 -1.3824
#>  -0.0885  3.3074  0.0844 -2.6239 -2.6705
#> 
#> (3,.,.) = 
#>  -0.4644 -4.3705  0.3642  2.3099 -1.3668
#>   1.2366 -5.7160  0.1704  3.2565 -3.0224
#>  -0.8045  3.4063  1.3238  1.0403  3.7934
#> 
#> (4,.,.) = 
#>   0.1573 -3.5709  2.7984 -1.4946 -2.5342
#>  -0.3741  0.0375  1.6970  0.9422  0.8038
#>   0.5169  0.4652 -1.2299  0.2455  2.2460
#> 
#> (5,.,.) = 
#>   3.1709  3.2819  2.2599  5.2354 -3.5856
#>  -2.7427 -0.7068  0.1890 -0.5377  0.7764
#>  -0.0118 -3.4196 -0.8138 -3.7205  1.4091
#> 
#> (6,.,.) = 
#>   3.8674 -2.0330 -1.1810 -4.5769 -0.9699
#>  -2.1033  3.8525  0.4855  3.4259 -1.6395
#>  -3.7183 -1.4197 -0.2313  1.2226  2.1281
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
