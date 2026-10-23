# Kullback-Leibler divergence loss

The Kullback-Leibler divergence loss measure Kullback-Leibler divergence
[doi:10.1214/aoms/1177729694](https://doi.org/10.1214/aoms/1177729694)
is a useful distance measure for continuous distributions and is often
useful when performing direct regression over the space of (discretely
sampled) continuous output distributions.

## Usage

``` r
nn_kl_div_loss(reduction = "mean")
```

## Arguments

- reduction:

  (string, optional): Specifies the reduction to apply to the output:
  `'none'` \| `'batchmean'` \| `'sum'` \| `'mean'`. `'none'`: no
  reduction will be applied. `'batchmean'`: the sum of the output will
  be divided by batchsize. `'sum'`: the output will be summed. `'mean'`:
  the output will be divided by the number of elements in the output.
  Default: `'mean'`

## Details

As with
[`nn_nll_loss()`](https://torch.mlverse.org/docs/dev/reference/nn_nll_loss.md),
the `input` given is expected to contain *log-probabilities* and is not
restricted to a 2D Tensor.

The targets are interpreted as *probabilities* by default, but could be
considered as *log-probabilities* with `log_target` set to `TRUE`.

This criterion expects a `target` `Tensor` of the same size as the
`input` `Tensor`.

The unreduced (i.e. with `reduction` set to `'none'`) loss can be
described as:

\$\$ l(x,y) = L = \\ l_1,\dots,l_N \\, \quad l_n = y_n \cdot \left( \log
y_n - x_n \right) \$\$

where the index \\N\\ spans all dimensions of `input` and \\L\\ has the
same shape as `input`. If `reduction` is not `'none'` (default
`'mean'`), then:

\$\$ \ell(x, y) = \begin{array}{ll} \mbox{mean}(L), & \mbox{if
reduction} = \mbox{'mean';} \\ \mbox{sum}(L), & \mbox{if reduction} =
\mbox{'sum'.} \end{array} \$\$

In default `reduction` mode `'mean'`, the losses are averaged for each
minibatch over observations **as well as** over dimensions.
`'batchmean'` mode gives the correct KL divergence where losses are
averaged over batch dimension only. `'mean'` mode's behavior will be
changed to the same as `'batchmean'` in the next major release.

## Note

`reduction` = `'mean'` doesn't return the true kl divergence value,
please use `reduction` = `'batchmean'` which aligns with KL math
definition. In the next major release, `'mean'` will be changed to be
the same as `'batchmean'`.

## Shape

- Input: \\(N, \*)\\ where \\\*\\ means, any number of additional
  dimensions

- Target: \\(N, \*)\\, same shape as the input

- Output: scalar by default. If `reduction` is `'none'`, then \\(N,
  \*)\\, the same shape as the input
