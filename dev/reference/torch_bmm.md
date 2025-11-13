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
#>   0.7481  0.4451 -0.7678  0.1655 -0.8834
#>   3.5177  0.6668  3.0350  3.4867 -3.9380
#>  -2.2879  0.4704 -0.8726 -0.9772 -1.0536
#> 
#> (2,.,.) = 
#>   0.0686 -0.1546  0.5987  0.4386  0.0572
#>   0.4699 -1.8164 -1.7383  2.2009 -0.6410
#>   0.8890  3.3681 -2.1927  0.9665 -1.7079
#> 
#> (3,.,.) = 
#>  -0.6605 -0.2847 -0.8480 -1.5756 -0.8688
#>   0.4576 -0.6067 -0.6938 -0.7963  0.0019
#>   0.7773 -1.3443 -2.2096 -7.3375 -1.0511
#> 
#> (4,.,.) = 
#>  -3.8451 -0.9882  1.7982 -0.6801 -4.2056
#>   0.1841  1.0135  2.2160 -0.2846 -0.8461
#>   1.0942  1.3159 -0.3433 -0.2772  0.8207
#> 
#> (5,.,.) = 
#>   3.8595 -0.2178  1.2161  0.4200 -2.2451
#>  -2.9457  2.1696 -3.6361  0.6037 -0.5209
#>  -1.6700  1.9918 -1.2385  0.9541 -1.7961
#> 
#> (6,.,.) = 
#>   0.0721 -1.5289  0.0535  3.4345 -1.6215
#>  -2.3237 -0.1319 -1.0624  5.2149 -0.6782
#>  -0.3811 -1.9136  0.5395  0.6628 -0.3650
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
