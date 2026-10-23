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
#>  5.3384  1.9135 -1.9344  1.9488  0.3851
#>   0.6814  1.5933  1.3873 -1.0139 -2.6648
#>   0.7535  0.7770 -0.4770  2.8327  1.0012
#> 
#> (2,.,.) = 
#>  0.7510  2.2508  0.6401  1.9289 -0.2074
#>   0.9018  0.5738 -1.0933  3.4280  1.9452
#>   0.2509  0.2354  0.2634 -0.6216 -2.3719
#> 
#> (3,.,.) = 
#>  1.1290  2.0410  0.1128 -4.8606  2.3235
#>   0.0202 -1.9491  7.6810 -0.2715  3.6062
#>  -0.0440 -0.2262 -5.6578  1.4393 -4.6288
#> 
#> (4,.,.) = 
#>  0.2941 -0.3321  1.3267  1.7915  0.7561
#>   2.1753  2.2708  1.6489 -1.7267  0.8531
#>  -0.8733  2.9209 -3.5499 -1.9046  0.8044
#> 
#> (5,.,.) = 
#>  2.7868  1.4691 -5.1734 -0.7097  0.1735
#>  -0.3361  1.8886  0.1420 -4.2988 -1.3481
#>   0.0529  0.4781  0.8006  1.0057 -0.3338
#> 
#> (6,.,.) = 
#>  0.9079 -0.4320 -1.8946  0.1119  3.3209
#>   0.3371  0.4863 -2.0170  2.8928  3.4391
#>   0.2733 -3.2355 -0.8305  2.5984 -0.1346
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{10,3,5} ]
```
