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
#>  -0.0138 -1.5639  1.6960 -1.1405  0.8079
#>  -1.0371 -2.7825 -1.2273 -0.1975 -1.2385
#>  -0.9389 -0.2999 -1.5357  1.4611 -0.6347
#> 
#> (2,.,.) = 
#>  -0.0810 -0.3336 -1.1633  1.1584 -0.8559
#>  -0.5197 -0.4143 -3.6801  4.5361 -2.8803
#>  -1.3800 -2.5635 -4.6151  4.8195 -4.3844
#> 
#> (3,.,.) = 
#>   2.3355 -0.2957  0.8646  1.1453  0.5900
#>  -0.1303  0.9075 -0.5817  0.8891 -1.5134
#>  -0.3699 -1.3144 -0.2725 -0.2885  0.7721
#> 
#> (4,.,.) = 
#>  -0.7564  1.2680 -3.2659  1.7543 -0.6335
#>  -1.1018 -0.4648  1.7731 -0.1835 -0.0547
#>   1.5408  1.4225  1.3658 -0.4481 -0.4880
#> 
#> (5,.,.) = 
#>  -2.4790 -1.7498  0.5300 -3.3235 -0.6285
#>   3.9119  5.5977 -0.9565  6.0159  0.2317
#>   0.0079 -0.0716 -0.0459 -0.6869  0.0303
#> 
#> (6,.,.) = 
#>  -1.0708 -1.8473 -0.0236 -1.8970 -1.6806
#>  -0.9045 -0.5080  0.4791 -3.3561 -2.2824
#>   0.7959  1.1164 -0.2606  2.1853  1.7674
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
