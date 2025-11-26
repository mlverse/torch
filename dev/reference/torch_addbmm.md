# Addbmm

Addbmm

## Usage

``` r
torch_addbmm(self, batch1, batch2, beta = 1L, alpha = 1L)
```

## Arguments

- self:

  (Tensor) matrix to be added

- batch1:

  (Tensor) the first batch of matrices to be multiplied

- batch2:

  (Tensor) the second batch of matrices to be multiplied

- beta:

  (Number, optional) multiplier for `input` (\\\beta\\)

- alpha:

  (Number, optional) multiplier for `batch1 @ batch2` (\\\alpha\\)

## addbmm(input, batch1, batch2, \*, beta=1, alpha=1, out=NULL) -\> Tensor

Performs a batch matrix-matrix product of matrices stored in `batch1`
and `batch2`, with a reduced add step (all matrix multiplications get
accumulated along the first dimension). `input` is added to the final
result.

`batch1` and `batch2` must be 3-D tensors each containing the same
number of matrices.

If `batch1` is a \\(b \times n \times m)\\ tensor, `batch2` is a \\(b
\times m \times p)\\ tensor, `input` must be broadcastable with a \\(n
\times p)\\ tensor and `out` will be a \\(n \times p)\\ tensor.

\$\$ out = \beta\\ \mbox{input} + \alpha\\ (\sum\_{i=0}^{b-1}
\mbox{batch1}\_i \mathbin{@} \mbox{batch2}\_i) \$\$ For inputs of type
`FloatTensor` or `DoubleTensor`, arguments `beta` and `alpha` must be
real numbers, otherwise they should be integers.

## Examples

``` r
if (torch_is_installed()) {

M = torch_randn(c(3, 5))
batch1 = torch_randn(c(10, 3, 4))
batch2 = torch_randn(c(10, 4, 5))
torch_addbmm(M, batch1, batch2)
}
#> torch_tensor
#>  10.3019   4.0364  -0.6847  -2.4041  -8.2855
#>  -3.2056   3.5002   2.8087   3.6494   4.5678
#> -12.6692  -1.2853   0.8850  -2.2896  -1.4250
#> [ CPUFloatType{3,5} ]
```
