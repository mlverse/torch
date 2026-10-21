# Baddbmm

Baddbmm

## Usage

``` r
torch_baddbmm(self, batch1, batch2, out_dtype, beta = 1L, alpha = 1L)
```

## Arguments

- self:

  (Tensor) the tensor to be added

- batch1:

  (Tensor) the first batch of matrices to be multiplied

- batch2:

  (Tensor) the second batch of matrices to be multiplied

- out_dtype:

  (torch_dtype, optional) the output dtype

- beta:

  (Number, optional) multiplier for `input` (\\\beta\\)

- alpha:

  (Number, optional) multiplier for \\\mbox{batch1} \mathbin{@}
  \mbox{batch2}\\ (\\\alpha\\)

## baddbmm(input, batch1, batch2, \*, beta=1, alpha=1, out=NULL) -\> Tensor

Performs a batch matrix-matrix product of matrices in `batch1` and
`batch2`. `input` is added to the final result.

`batch1` and `batch2` must be 3-D tensors each containing the same
number of matrices.

If `batch1` is a \\(b \times n \times m)\\ tensor, `batch2` is a \\(b
\times m \times p)\\ tensor, then `input` must be broadcastable with a
\\(b \times n \times p)\\ tensor and `out` will be a \\(b \times n
\times p)\\ tensor. Both `alpha` and `beta` mean the same as the scaling
factors used in `torch_addbmm`.

\$\$ \mbox{out}\_i = \beta\\ \mbox{input}\_i + \alpha\\
(\mbox{batch1}\_i \mathbin{@} \mbox{batch2}\_i) \$\$ For inputs of type
`FloatTensor` or `DoubleTensor`, arguments `beta` and `alpha` must be
real numbers, otherwise they should be integers.

## Examples

``` r
if (torch_is_installed()) {

M = torch_randn(c(10, 3, 5))
batch1 = torch_randn(c(10, 3, 4))
batch2 = torch_randn(c(10, 4, 5))
torch_baddbmm(M, batch1, batch2)
}
#> torch_tensor
#> (1,.,.) = 
#> -0.3255 -1.3819  1.4254 -0.3955 -0.2669
#>   1.5926 -0.8978  0.1848 -2.6307 -0.3083
#>  -1.6084 -0.5088 -1.8487 -1.3773 -0.0088
#> 
#> (2,.,.) = 
#>  3.7293  1.0426  0.5020 -2.2426 -1.4184
#>  -6.1610 -0.6442  0.3002  3.0515 -1.9869
#>   1.8277  4.7234  0.5714  0.4490  0.5752
#> 
#> (3,.,.) = 
#> -0.1016  0.1883 -1.6377 -6.3652  0.7608
#>  -0.6011  2.5685  0.1373 -0.4387 -3.8402
#>   0.4514 -0.4007 -2.1581 -6.1963  3.0306
#> 
#> (4,.,.) = 
#> -2.6862 -2.8212  1.8259 -1.3442  1.7917
#>  -0.7387  1.8874  1.6535  0.6120  0.0345
#>   3.3585  2.2329  3.2081  1.0364 -0.7462
#> 
#> (5,.,.) = 
#> -1.3514  2.5893 -0.9465  0.6600  1.9785
#>   2.8574  2.4144 -3.1503 -2.4203 -0.8469
#>   0.2409 -0.0820 -0.5947 -0.8334  3.4449
#> 
#> (6,.,.) = 
#> -1.6255 -0.4532 -2.9873  2.3951 -0.7968
#>   1.6863  1.3161  1.6492 -0.2836 -0.6403
#>  -1.3994 -0.5765  2.8440 -1.5983 -0.0293
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
