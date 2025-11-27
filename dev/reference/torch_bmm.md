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
#>  -3.1486  1.4221  3.1676  2.9599  1.9272
#>   0.6231  0.6259  0.5040 -0.0462  0.4790
#>   0.7363  0.4433 -0.0538 -0.2339  1.6395
#> 
#> (2,.,.) = 
#>   1.1059 -0.0261 -4.1729 -0.6737  0.6757
#>   2.7215  1.0808  2.3339 -1.7919  0.6156
#>  -2.4005 -2.1016 -1.4336  1.9928 -1.5641
#> 
#> (3,.,.) = 
#>  -0.7406  0.8288  2.6789 -0.1191 -0.5894
#>   0.4311 -5.5646 -0.3951 -0.4351 -2.9950
#>  -2.0241  0.0363 -1.3463 -0.7737  1.6244
#> 
#> (4,.,.) = 
#>  -2.1288 -0.9217 -2.6612 -0.7730 -2.7714
#>   0.7157  1.7880  0.6882  0.6189  1.0815
#>  -0.0037 -2.9816 -1.3016 -0.6419 -0.5553
#> 
#> (5,.,.) = 
#>  -0.1518  0.3939  0.1188 -0.4019  0.2272
#>   1.6012  0.8364 -0.3868  0.5572 -0.3627
#>  -1.3724 -0.4708 -1.1312 -2.8950 -0.9013
#> 
#> (6,.,.) = 
#>   1.0112 -1.4352 -0.6020 -2.2624  0.6755
#>   1.4140  3.4254 -0.5567  2.1072 -0.2770
#>  -1.3107 -2.6632 -1.8468 -0.8434 -0.2513
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
