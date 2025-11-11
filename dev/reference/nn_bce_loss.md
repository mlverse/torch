# Binary cross entropy loss

Creates a criterion that measures the Binary Cross Entropy between the
target and the output:

## Usage

``` r
nn_bce_loss(weight = NULL, reduction = "mean")
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

## Details

The unreduced (i.e. with `reduction` set to `'none'`) loss can be
described as: \$\$ \ell(x, y) = L = \\l_1,\dots,l_N\\^\top, \quad l_n
= - w_n \left\[ y_n \cdot \log x_n + (1 - y_n) \cdot \log (1 - x_n)
\right\] \$\$ where \\N\\ is the batch size. If `reduction` is not
`'none'` (default `'mean'`), then

\$\$ \ell(x, y) = \left\\ \begin{array}{ll} \mbox{mean}(L), & \mbox{if
reduction} = \mbox{'mean';}\\ \mbox{sum}(L), & \mbox{if reduction} =
\mbox{'sum'.} \end{array} \right. \$\$

This is used for measuring the error of a reconstruction in for example
an auto-encoder. Note that the targets \\y\\ should be numbers between 0
and 1.

Notice that if \\x_n\\ is either 0 or 1, one of the log terms would be
mathematically undefined in the above loss equation. PyTorch chooses to
set \\\log (0) = -\infty\\, since \\\lim\_{x\to 0} \log (x) = -\infty\\.

However, an infinite term in the loss equation is not desirable for
several reasons. For one, if either \\y_n = 0\\ or \\(1 - y_n) = 0\\,
then we would be multiplying 0 with infinity. Secondly, if we have an
infinite loss value, then we would also have an infinite term in our
gradient, since \\\lim\_{x\to 0} \frac{d}{dx} \log (x) = \infty\\.

This would make BCELoss's backward method nonlinear with respect to
\\x_n\\, and using it for things like linear regression would not be
straight-forward. Our solution is that BCELoss clamps its log function
outputs to be greater than or equal to -100. This way, we can always
have a finite loss value and a linear backward method.

## Shape

- Input: \\(N, \*)\\ where \\\*\\ means, any number of additional
  dimensions

- Target: \\(N, \*)\\, same shape as the input

- Output: scalar. If `reduction` is `'none'`, then \\(N, \*)\\, same
  shape as input.

## Examples

``` r
if (torch_is_installed()) {
m <- nn_sigmoid()
loss <- nn_bce_loss()
input <- torch_randn(3, requires_grad = TRUE)
target <- torch_rand(3)
output <- loss(m(input), target)
output$backward()
}
```
