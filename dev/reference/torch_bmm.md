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
#>  -2.0670 -0.2925 -0.3676 -1.7161  2.4606
#>  -0.3552 -0.1254 -0.8835 -1.6975  3.4306
#>  -2.8334 -1.3176 -0.0784 -3.0826  2.4835
#> 
#> (2,.,.) = 
#>   0.3208  0.1319  0.7022 -0.5237 -0.2966
#>  -1.1570 -2.0641  3.3727 -2.1404 -1.5586
#>   2.3327 -0.4389  1.6121  1.7007 -0.5032
#> 
#> (3,.,.) = 
#>   1.9272  1.1813  1.6356  3.8096 -4.6018
#>  -0.9332 -2.5773  1.3280  0.5967  1.9724
#>   0.4482 -1.3119  1.2149  0.1334 -3.7545
#> 
#> (4,.,.) = 
#>   0.0389 -2.7414 -0.5285  2.1778  1.7103
#>   1.6998  1.7364  0.8507 -0.6058 -0.6120
#>  -0.8306 -0.9621 -0.4782  1.2349  1.1689
#> 
#> (5,.,.) = 
#>   0.0170  0.5967  0.1822 -0.3238 -1.7107
#>  -0.9185  1.5143  1.8834  0.7444 -0.2201
#>  -0.0364  0.9953  0.0970 -1.2529 -2.5461
#> 
#> (6,.,.) = 
#>   5.9386 -0.9429 -2.7082  1.1253 -2.2188
#>   7.4100 -3.4717 -3.1876  2.1133 -0.7648
#>   2.7674 -2.0707 -2.0075  3.0732  0.9979
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
