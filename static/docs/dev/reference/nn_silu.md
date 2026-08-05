# Applies the Sigmoid Linear Unit (SiLU) function, element-wise. The SiLU function is also known as the swish function.

Applies the Sigmoid Linear Unit (SiLU) function, element-wise. The SiLU
function is also known as the swish function.

## Usage

``` r
nn_silu(inplace = FALSE)
```

## Arguments

- inplace:

  can optionally do the operation in-place. Default: `FALSE`

## Details

See [Gaussian Error Linear Units
(GELUs)](https://arxiv.org/abs/1606.08415) where the SiLU (Sigmoid
Linear Unit) was originally coined, and see [Sigmoid-Weighted Linear
Units for Neural Network Function Approximation in Reinforcement
Learning](https://arxiv.org/abs/1702.03118) and [Swish: a Self-Gated
Activation Function](https://arxiv.org/abs/1710.05941v1) where the SiLU
was experimented with later.
