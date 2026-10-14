# Poisson NLL loss

Negative log likelihood loss with Poisson distribution of target. The
loss can be described as:

## Usage

``` r
nn_poisson_nll_loss(
  log_input = TRUE,
  full = FALSE,
  eps = 1e-08,
  reduction = "mean"
)
```

## Arguments

- log_input:

  (bool, optional): if `TRUE` the loss is computed as
  \\\exp(\mbox{input}) - \mbox{target}\*\mbox{input}\\, if `FALSE` the
  loss is \\\mbox{input} -
  \mbox{target}\*\log(\mbox{input}+\mbox{eps})\\.

- full:

  (bool, optional): whether to compute full loss, i. e. to add the
  Stirling approximation term \\\mbox{target}\*\log(\mbox{target}) -
  \mbox{target} + 0.5 \* \log(2\pi\mbox{target})\\.

- eps:

  (float, optional): Small value to avoid evaluation of \\\log(0)\\ when
  `log_input = FALSE`. Default: 1e-8

- reduction:

  (string, optional): Specifies the reduction to apply to the output:
  `'none'` \| `'mean'` \| `'sum'`. `'none'`: no reduction will be
  applied, `'mean'`: the sum of the output will be divided by the number
  of elements in the output, `'sum'`: the output will be summed.

## Details

\$\$ \mbox{target} \sim \mathrm{Poisson}(\mbox{input})
\mbox{loss}(\mbox{input}, \mbox{target}) = \mbox{input} - \mbox{target}
\* \log(\mbox{input}) + \log(\mbox{target!}) \$\$

The last term can be omitted or approximated with Stirling formula. The
approximation is used for target values more than 1. For targets less or
equal to 1 zeros are added to the loss.

## Shape

- Input: \\(N, \*)\\ where \\\*\\ means, any number of additional
  dimensions

- Target: \\(N, \*)\\, same shape as the input

- Output: scalar by default. If `reduction` is `'none'`, then \\(N,
  \*)\\, the same shape as the input

## Examples

``` r
if (torch_is_installed()) {
loss <- nn_poisson_nll_loss()
log_input <- torch_randn(5, 2, requires_grad = TRUE)
target <- torch_randn(5, 2)
output <- loss(log_input, target)
output$backward()
}
```
