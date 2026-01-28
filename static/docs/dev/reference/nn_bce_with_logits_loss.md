# BCE with logits loss

This loss combines a `Sigmoid` layer and the `BCELoss` in one single
class. This version is more numerically stable than using a plain
`Sigmoid` followed by a `BCELoss` as, by combining the operations into
one layer, we take advantage of the log-sum-exp trick for numerical
stability.

## Usage

``` r
nn_bce_with_logits_loss(weight = NULL, reduction = "mean", pos_weight = NULL)
```

## Arguments

- weight:

  (Tensor, optional): a manual rescaling weight given to the loss of
  each batch element. If given, has to be a Tensor of size `nbatch`.

- reduction:

  (string, optional): Specifies the reduction to apply to the output:
  `'none'` \| `'mean'` \| `'sum'`. `'none'`: no reduction will be
  applied, `'mean'`: the sum of the output will be divided by the number
  of elements in the output, `'sum'`: the output will be summed.

- pos_weight:

  (Tensor, optional): a weight of positive examples. Must be a vector
  with length equal to the number of classes.

## Details

The unreduced (i.e. with `reduction` set to `'none'`) loss can be
described as:

\$\$ \ell(x, y) = L = \\l_1,\dots,l_N\\^\top, \quad l_n = - w_n \left\[
y_n \cdot \log \sigma(x_n) + (1 - y_n) \cdot \log (1 - \sigma(x_n))
\right\], \$\$

where \\N\\ is the batch size. If `reduction` is not `'none'` (default
`'mean'`), then

\$\$ \ell(x, y) = \begin{array}{ll} \mbox{mean}(L), & \mbox{if
reduction} = \mbox{'mean';}\\ \mbox{sum}(L), & \mbox{if reduction} =
\mbox{'sum'.} \end{array} \$\$

This is used for measuring the error of a reconstruction in for example
an auto-encoder. Note that the targets `t[i]` should be numbers between
0 and 1. It's possible to trade off recall and precision by adding
weights to positive examples. In the case of multi-label classification
the loss can be described as:

\$\$ \ell_c(x, y) = L_c = \\l\_{1,c},\dots,l\_{N,c}\\^\top, \quad
l\_{n,c} = - w\_{n,c} \left\[ p_c y\_{n,c} \cdot \log \sigma(x\_{n,c}) +
(1 - y\_{n,c}) \cdot \log (1 - \sigma(x\_{n,c})) \right\], \$\$ where
\\c\\ is the class number (\\c \> 1\\ for multi-label binary
classification,

\\c = 1\\ for single-label binary classification), \\n\\ is the number
of the sample in the batch and \\p_c\\ is the weight of the positive
answer for the class \\c\\. \\p_c \> 1\\ increases the recall, \\p_c \<
1\\ increases the precision. For example, if a dataset contains 100
positive and 300 negative examples of a single class, then `pos_weight`
for the class should be equal to \\\frac{300}{100}=3\\. The loss would
act as if the dataset contains \\3\times 100=300\\ positive examples.

## Shape

- Input: \\(N, \*)\\ where \\\*\\ means, any number of additional
  dimensions

- Target: \\(N, \*)\\, same shape as the input

- Output: scalar. If `reduction` is `'none'`, then \\(N, \*)\\, same
  shape as input.

## Examples

``` r
if (torch_is_installed()) {
loss <- nn_bce_with_logits_loss()
input <- torch_randn(3, requires_grad = TRUE)
target <- torch_empty(3)$random_(1, 2)
output <- loss(input, target)
output$backward()

target <- torch_ones(10, 64, dtype = torch_float32()) # 64 classes, batch size = 10
output <- torch_full(c(10, 64), 1.5) # A prediction (logit)
pos_weight <- torch_ones(64) # All weights are equal to 1
criterion <- nn_bce_with_logits_loss(pos_weight = pos_weight)
criterion(output, target) # -log(sigmoid(1.5))
}
#> torch_tensor
#> 0.201413
#> [ CPUFloatType{} ]
```
