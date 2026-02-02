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
#>   1.8708  0.0919 -0.6600  5.9985  2.8642
#>   1.1409 -0.1575 -0.2228  0.0357  1.9521
#>   0.0245 -0.1080 -0.7209  2.8656  1.3702
#> 
#> (2,.,.) = 
#>  -0.7021 -0.2271 -2.9862 -2.1664  0.8602
#>  -2.2556 -0.4748  3.4066  1.1110 -1.5317
#>   0.1933  1.2604 -0.6901  1.4060 -0.9551
#> 
#> (3,.,.) = 
#>   0.8130  2.0010 -1.0524 -0.1067 -0.0077
#>  -0.4705  0.0127 -1.9303 -1.1477 -1.2649
#>  -1.3186 -1.4989 -2.4263 -2.2355 -0.6995
#> 
#> (4,.,.) = 
#>   1.1262  1.8197 -1.7909 -1.5912  0.5170
#>   1.7681  2.2362  0.6289 -0.5368  1.2108
#>   0.1418 -2.4852 -1.2916 -0.0940  1.0653
#> 
#> (5,.,.) = 
#>  -0.6322  2.6563  1.0329 -0.5707  0.7737
#>   0.3387  4.9575  0.3933  4.4707 -0.8505
#>  -1.3541 -2.1312 -0.1712  1.0436  0.0844
#> 
#> (6,.,.) = 
#>   0.0094  0.3153 -1.5465  0.9911 -2.3895
#>  -1.1360 -0.3970 -0.4616  1.1277 -0.0550
#>  -3.2429 -1.0083 -1.2964 -0.9492 -0.9345
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
