# Group normalization

Applies Group Normalization over a mini-batch of inputs as described in
the paper [Group Normalization](https://arxiv.org/abs/1803.08494).

## Usage

``` r
nn_group_norm(num_groups, num_channels, eps = 1e-05, affine = TRUE)
```

## Arguments

- num_groups:

  (int): number of groups to separate the channels into

- num_channels:

  (int): number of channels expected in input

- eps:

  a value added to the denominator for numerical stability. Default:
  1e-5

- affine:

  a boolean value that when set to `TRUE`, this module has learnable
  per-channel affine parameters initialized to ones (for weights) and
  zeros (for biases). Default: `TRUE`.

## Details

\$\$ y = \frac{x - \mathrm{E}\[x\]}{ \sqrt{\mathrm{Var}\[x\] +
\epsilon}} \* \gamma + \beta \$\$

The input channels are separated into `num_groups` groups, each
containing `num_channels / num_groups` channels. The mean and
standard-deviation are calculated separately over the each group.
\\\gamma\\ and \\\beta\\ are learnable per-channel affine transform
parameter vectors of size `num_channels` if `affine` is `TRUE`. The
standard-deviation is calculated via the biased estimator, equivalent to
`torch_var(input, unbiased=FALSE)`.

## Note

This layer uses statistics computed from input data in both training and
evaluation modes.

## Shape

- Input: \\(N, C, \*)\\ where \\C=\mbox{num\\channels}\\

- Output: \\(N, C, \*)\\\` (same shape as input)

## Examples

``` r
if (torch_is_installed()) {

input <- torch_randn(20, 6, 10, 10)
# Separate 6 channels into 3 groups
m <- nn_group_norm(3, 6)
# Separate 6 channels into 6 groups (equivalent with [nn_instance_morm])
m <- nn_group_norm(6, 6)
# Put all 6 channels into a single group (equivalent with [nn_layer_norm])
m <- nn_group_norm(1, 6)
# Activating the module
output <- m(input)
}
```
