# Multilabel margin loss

Creates a criterion that optimizes a multi-class multi-classification
hinge loss (margin-based loss) between input \\x\\ (a 2D mini-batch
`Tensor`) and output \\y\\ (which is a 2D `Tensor` of target class
indices). For each sample in the mini-batch:

## Usage

``` r
nn_multilabel_margin_loss(reduction = "mean")
```

## Arguments

- reduction:

  (string, optional): Specifies the reduction to apply to the output:
  `'none'` \| `'mean'` \| `'sum'`. `'none'`: no reduction will be
  applied, `'mean'`: the sum of the output will be divided by the number
  of elements in the output, `'sum'`: the output will be summed.

## Details

\$\$ \mbox{loss}(x, y) = \sum\_{ij}\frac{\max(0, 1 - (x\[y\[j\]\] -
x\[i\]))}{\mbox{x.size}(0)} \$\$

where \\x \in \left\\0, \\ \cdots , \\ \mbox{x.size}(0) - 1\right\\\\,
\\ \\y \in \left\\0, \\ \cdots , \\ \mbox{y.size}(0) - 1\right\\\\, \\
\\0 \leq y\[j\] \leq \mbox{x.size}(0)-1\\, \\ and \\i \neq y\[j\]\\ for
all \\i\\ and \\j\\. \\y\\ and \\x\\ must have the same size.

The criterion only considers a contiguous block of non-negative targets
that starts at the front. This allows for different samples to have
variable amounts of target classes.

## Shape

- Input: \\(C)\\ or \\(N, C)\\ where `N` is the batch size and `C` is
  the number of classes.

- Target: \\(C)\\ or \\(N, C)\\, label targets padded by -1 ensuring
  same shape as the input.

- Output: scalar. If `reduction` is `'none'`, then \\(N)\\.

## Examples

``` r
if (torch_is_installed()) {
loss <- nn_multilabel_margin_loss()
x <- torch_tensor(c(0.1, 0.2, 0.4, 0.8))$view(c(1, 4))
# for target y, only consider labels 4 and 1, not after label -1
y <- torch_tensor(c(4, 1, -1, 2), dtype = torch_long())$view(c(1, 4))
loss(x, y)
}
#> torch_tensor
#> 0.85
#> [ CPUFloatType{} ]
```
