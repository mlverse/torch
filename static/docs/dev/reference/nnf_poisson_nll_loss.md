# Poisson_nll_loss

Poisson negative log likelihood loss.

## Usage

``` r
nnf_poisson_nll_loss(
  input,
  target,
  log_input = TRUE,
  full = FALSE,
  eps = 1e-08,
  reduction = "mean"
)
```

## Arguments

- input:

  tensor (N,\*) where \*\* means, any number of additional dimensions

- target:

  tensor (N,\*) , same shape as the input

- log_input:

  if `TRUE` the loss is computed as \\\exp(\mbox{input}) - \mbox{target}
  \* \mbox{input}\\, if `FALSE` then loss is \\\mbox{input} -
  \mbox{target} \* \log(\mbox{input}+\mbox{eps})\\. Default: `TRUE`.

- full:

  whether to compute full loss, i. e. to add the Stirling approximation
  term. Default: `FALSE`.

- eps:

  (float, optional) Small value to avoid evaluation of \\\log(0)\\ when
  `log_input`=`FALSE`. Default: 1e-8

- reduction:

  (string, optional) – Specifies the reduction to apply to the output:
  'none' \| 'mean' \| 'sum'. 'none': no reduction will be applied,
  'mean': the sum of the output will be divided by the number of
  elements in the output, 'sum': the output will be summed. Default:
  'mean'
