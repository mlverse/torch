# Dropout module

During training, randomly zeroes some of the elements of the input
tensor with probability `p` using samples from a Bernoulli distribution.
Each channel will be zeroed out independently on every forward call.

## Usage

``` r
nn_dropout(p = 0.5, inplace = FALSE)
```

## Arguments

- p:

  probability of an element to be zeroed. Default: 0.5

- inplace:

  If set to `TRUE`, will do this operation in-place. Default: `FALSE`.

## Details

This has proven to be an effective technique for regularization and
preventing the co-adaptation of neurons as described in the paper
[Improving neural networks by preventing co-adaptation of feature
detectors](https://arxiv.org/abs/1207.0580).

Furthermore, the outputs are scaled by a factor of :math:`\frac{1}{1-p}`
during training. This means that during evaluation the module simply
computes an identity function.

## Shape

- Input: \\(\*)\\. Input can be of any shape

- Output: \\(\*)\\. Output is of the same shape as input

## Examples

``` r
if (torch_is_installed()) {
m <- nn_dropout(p = 0.2)
input <- torch_randn(20, 16)
output <- m(input)
}
```
