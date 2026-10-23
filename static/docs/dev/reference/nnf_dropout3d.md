# Dropout3d

Randomly zero out entire channels (a channel is a 3D feature map, e.g.,
the \\j\\-th channel of the \\i\\-th sample in the batched input is a 3D
tensor \\input\[i, j\]\\) of the input tensor). Each channel will be
zeroed out independently on every forward call with probability `p`
using samples from a Bernoulli distribution.

## Usage

``` r
nnf_dropout3d(input, p = 0.5, training = TRUE, inplace = FALSE)
```

## Arguments

- input:

  the input tensor

- p:

  probability of a channel to be zeroed. Default: 0.5

- training:

  apply dropout if is `TRUE`. Default: `TRUE`.

- inplace:

  If set to `TRUE`, will do this operation in-place. Default: `FALSE`
