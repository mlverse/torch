# L1 loss

Creates a criterion that measures the mean absolute error (MAE) between
each element in the input \\x\\ and target \\y\\.

## Usage

``` r
nn_l1_loss(reduction = "mean")
```

## Arguments

- reduction:

  (string, optional): Specifies the reduction to apply to the output:
  `'none'` \| `'mean'` \| `'sum'`. `'none'`: no reduction will be
  applied, `'mean'`: the sum of the output will be divided by the number
  of elements in the output, `'sum'`: the output will be summed.

## Details

The unreduced (i.e. with `reduction` set to `'none'`) loss can be
described as:

\$\$ \ell(x, y) = L = \\l_1,\dots,l_N\\^\top, \quad l_n = \left\| x_n -
y_n \right\|, \$\$

where \\N\\ is the batch size. If `reduction` is not `'none'` (default
`'mean'`), then:

\$\$ \ell(x, y) = \begin{array}{ll} \mbox{mean}(L), & \mbox{if
reduction} = \mbox{'mean';}\\ \mbox{sum}(L), & \mbox{if reduction} =
\mbox{'sum'.} \end{array} \$\$

\\x\\ and \\y\\ are tensors of arbitrary shapes with a total of \\n\\
elements each.

The sum operation still operates over all the elements, and divides by
\\n\\. The division by \\n\\ can be avoided if one sets
`reduction = 'sum'`.

## Shape

- Input: \\(N, \*)\\ where \\\*\\ means, any number of additional
  dimensions

- Target: \\(N, \*)\\, same shape as the input

- Output: scalar. If `reduction` is `'none'`, then \\(N, \*)\\, same
  shape as the input

## Examples

``` r
if (torch_is_installed()) {
loss <- nn_l1_loss()
input <- torch_randn(3, 5, requires_grad = TRUE)
target <- torch_randn(3, 5)
output <- loss(input, target)
output$backward()
}
```
