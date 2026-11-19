# BatchNorm1D module

Applies Batch Normalization over a 2D or 3D input (a mini-batch of 1D
inputs with optional additional channel dimension) as described in the
paper [Batch Normalization: Accelerating Deep Network Training by
Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)

## Usage

``` r
nn_batch_norm1d(
  num_features,
  eps = 1e-05,
  momentum = 0.1,
  affine = TRUE,
  track_running_stats = TRUE
)
```

## Arguments

- num_features:

  \\C\\ from an expected input of size \\(N, C, L)\\ or \\L\\ from input
  of size \\(N, L)\\

- eps:

  a value added to the denominator for numerical stability. Default:
  1e-5

- momentum:

  the value used for the running_mean and running_var computation. Can
  be set to `NULL` for cumulative moving average (i.e. simple average).
  Default: 0.1

- affine:

  a boolean value that when set to `TRUE`, this module has learnable
  affine parameters. Default: `TRUE`

- track_running_stats:

  a boolean value that when set to `TRUE`, this module tracks the
  running mean and variance, and when set to `FALSE`, this module does
  not track such statistics and always uses batch statistics in both
  training and eval modes. Default: `TRUE`

## Details

\$\$ y = \frac{x - \mathrm{E}\[x\]}{\sqrt{\mathrm{Var}\[x\] + \epsilon}}
\* \gamma + \beta \$\$

The mean and standard-deviation are calculated per-dimension over the
mini-batches and \\\gamma\\ and \\\beta\\ are learnable parameter
vectors of size `C` (where `C` is the input size). By default, the
elements of \\\gamma\\ are set to 1 and the elements of \\\beta\\ are
set to 0.

Also by default, during training this layer keeps running estimates of
its computed mean and variance, which are then used for normalization
during evaluation. The running estimates are kept with a default
:attr:`momentum` of 0.1. If `track_running_stats` is set to `FALSE`,
this layer then does not keep running estimates, and batch statistics
are instead used during evaluation time as well.

## Note

This `momentum` argument is different from one used in optimizer classes
and the conventional notion of momentum. Mathematically, the update rule
for running statistics here is \\\hat{x}\_{\mbox{new}} = (1 -
\mbox{momentum}) \times \hat{x} + \mbox{momentum} \times x_t\\, where
\\\hat{x}\\ is the estimated statistic and \\x_t\\ is the new observed
value.

Because the Batch Normalization is done over the `C` dimension,
computing statistics on `(N, L)` slices, it's common terminology to call
this Temporal Batch Normalization.

## Shape

- Input: \\(N, C)\\ or \\(N, C, L)\\

- Output: \\(N, C)\\ or \\(N, C, L)\\ (same shape as input)

## Examples

``` r
if (torch_is_installed()) {
# With Learnable Parameters
m <- nn_batch_norm1d(100)
# Without Learnable Parameters
m <- nn_batch_norm1d(100, affine = FALSE)
input <- torch_randn(20, 100)
output <- m(input)
}
```
