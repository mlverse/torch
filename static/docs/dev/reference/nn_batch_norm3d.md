# BatchNorm3D

Applies Batch Normalization over a 5D input (a mini-batch of 3D inputs
with additional channel dimension) as described in the paper [Batch
Normalization: Accelerating Deep Network Training by Reducing Internal
Covariate Shift](https://arxiv.org/abs/1502.03167).

## Usage

``` r
nn_batch_norm3d(
  num_features,
  eps = 1e-05,
  momentum = 0.1,
  affine = TRUE,
  track_running_stats = TRUE
)
```

## Arguments

- num_features:

  \\C\\ from an expected input of size \\(N, C, D, H, W)\\

- eps:

  a value added to the denominator for numerical stability. Default:
  1e-5

- momentum:

  the value used for the running_mean and running_var computation. Can
  be set to `None` for cumulative moving average (i.e. simple average).
  Default: 0.1

- affine:

  a boolean value that when set to `TRUE`, this module has learnable
  affine parameters. Default: `TRUE`

- track_running_stats:

  a boolean value that when set to `TRUE`, this module tracks the
  running mean and variance, and when set to `FALSE`, this module does
  not track such statistics and uses batch statistics instead in both
  training and eval modes if the running mean and variance are `None`.
  Default: `TRUE`

## Details

\$\$ y = \frac{x - \mathrm{E}\[x\]}{ \sqrt{\mathrm{Var}\[x\] +
\epsilon}} \* \gamma + \beta \$\$

The mean and standard-deviation are calculated per-dimension over the
mini-batches and \\\gamma\\ and \\\beta\\ are learnable parameter
vectors of size `C` (where `C` is the input size). By default, the
elements of \\\gamma\\ are set to 1 and the elements of \\\beta\\ are
set to 0. The standard-deviation is calculated via the biased estimator,
equivalent to `torch_var(input, unbiased = FALSE)`.

Also by default, during training this layer keeps running estimates of
its computed mean and variance, which are then used for normalization
during evaluation. The running estimates are kept with a default
`momentum` of 0.1.

If `track_running_stats` is set to `FALSE`, this layer then does not
keep running estimates, and batch statistics are instead used during
evaluation time as well.

## Note

This `momentum` argument is different from one used in optimizer classes
and the conventional notion of momentum. Mathematically, the update rule
for running statistics here is: \\\hat{x}\_{\mbox{new}} = (1 -
\mbox{momentum}) \times \hat{x} + \mbox{momentum} \times x_t\\, where
\\\hat{x}\\ is the estimated statistic and \\x_t\\ is the new observed
value.

Because the Batch Normalization is done over the `C` dimension,
computing statistics on `(N, D, H, W)` slices, it's common terminology
to call this Volumetric Batch Normalization or Spatio-temporal Batch
Normalization.

## Shape

- Input: \\(N, C, D, H, W)\\

- Output: \\(N, C, D, H, W)\\ (same shape as input)

## Examples

``` r
if (torch_is_installed()) {
# With Learnable Parameters
m <- nn_batch_norm3d(100)
# Without Learnable Parameters
m <- nn_batch_norm3d(100, affine = FALSE)
input <- torch_randn(20, 100, 35, 45, 55)
output <- m(input)
}
```
