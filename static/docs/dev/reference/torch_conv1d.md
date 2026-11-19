# Conv1d

Conv1d

## Usage

``` r
torch_conv1d(
  input,
  weight,
  bias = list(),
  stride = 1L,
  padding = 0L,
  dilation = 1L,
  groups = 1L
)
```

## Arguments

- input:

  input tensor of shape \\(\mbox{minibatch} , \mbox{in\\channels} ,
  iW)\\

- weight:

  filters of shape \\(\mbox{out\\channels} ,
  \frac{\mbox{in\\channels}}{\mbox{groups}} , kW)\\

- bias:

  optional bias of shape \\(\mbox{out\\channels})\\. Default: `NULL`

- stride:

  the stride of the convolving kernel. Can be a single number or a
  one-element tuple `(sW,)`. Default: 1

- padding:

  implicit paddings on both sides of the input. Can be a single number
  or a one-element tuple `(padW,)`. Default: 0

- dilation:

  the spacing between kernel elements. Can be a single number or a
  one-element tuple `(dW,)`. Default: 1

- groups:

  split input into groups, \\\mbox{in\\channels}\\ should be divisible
  by the number of groups. Default: 1

## conv1d(input, weight, bias=NULL, stride=1, padding=0, dilation=1, groups=1) -\> Tensor

Applies a 1D convolution over an input signal composed of several input
planes.

See
[`nn_conv1d()`](https://torch.mlverse.org/docs/dev/reference/nn_conv1d.md)
for details and output shape.

## Examples

``` r
if (torch_is_installed()) {

filters = torch_randn(c(33, 16, 3))
inputs = torch_randn(c(20, 16, 50))
nnf_conv1d(inputs, filters)
}
#> torch_tensor
#> (1,.,.) = 
#> Columns 1 to 6 -3.9654e-02  5.2799e+00 -6.8852e-01 -7.5415e+00  1.8799e+00  8.5805e-01
#>  -2.2126e+00  9.1994e+00 -4.5385e+00  1.7506e+00  5.5899e+00  9.0272e+00
#>  -1.7917e-01 -6.0686e+00 -5.0292e+00 -1.2820e+01  8.3589e+00 -8.1823e+00
#>  -4.1287e-02 -2.7174e+00  4.6697e+00 -4.4490e+00  7.4071e+00 -1.6472e+00
#>  -4.2357e+00  8.1810e+00 -8.7006e+00 -9.8603e+00 -2.6485e+00 -1.0556e+01
#>  -9.9692e+00 -1.7946e+01 -1.4545e+01 -4.9311e+00  9.1009e+00 -5.2455e+00
#>  -2.6934e+00 -2.4323e+00  1.0012e+01  1.2403e+01 -1.0677e+01  8.7325e+00
#>  -4.2566e+00  1.0414e+01 -6.0598e+00 -7.4569e+00 -2.7381e+00 -2.6915e+00
#>   2.0054e+00  3.2601e+00  1.2903e+00 -4.0800e+00  5.0983e+00  2.5374e+00
#>  -1.1924e+01 -7.7884e+00  1.5174e+00  9.7585e+00 -4.3451e-01 -2.8914e-01
#>   4.3538e+00  2.5390e+00  2.2765e+01 -2.0847e+01  6.5922e+00  4.7134e+00
#>   1.1311e+00 -2.3589e+00 -3.1082e+00  6.5878e+00  7.5254e+00  3.9054e+00
#>   1.4023e+00  1.0307e+01 -1.2420e+00 -7.0779e+00  6.5990e-01  5.0998e-01
#>   2.4965e+00  4.6327e-01  5.6820e+00 -4.3500e-01  9.9877e+00  2.4150e+00
#>  -4.1609e+00 -1.1321e+01 -3.7353e+00  1.2815e+01 -1.0037e+01  5.1755e+00
#>  -1.3866e+00  8.9021e-02  4.1274e+00  4.1063e+00 -2.0409e+00  3.3421e-01
#>   3.1718e+00 -1.2506e+00 -6.1100e+00  1.1724e+01  7.2672e+00  1.7272e+00
#>  -3.2257e+00 -1.3332e+01  5.0389e+00  1.1767e+01  7.0173e+00 -8.8269e+00
#>   7.6479e+00  6.6587e+00  3.1631e+00  4.0623e+00  8.7575e+00 -2.8879e+00
#>  -7.0411e+00 -6.5051e-01  5.0800e+00 -1.2043e+00  8.6294e+00  6.9444e+00
#>  -1.2743e+00  7.4096e+00 -1.7506e-01 -1.8464e+00 -6.0909e+00 -4.0320e+00
#>   5.2603e+00  3.6716e-01  1.3104e+00  6.0468e+00 -1.3489e+01 -1.0245e+01
#>   4.5086e+00 -3.3626e+00  7.5175e+00  9.6927e+00  1.4783e-01  1.3648e+01
#>   5.9148e-01 -8.8070e-01 -5.1688e+00 -9.6386e+00  1.3915e+01 -9.2702e+00
#>  -6.2738e+00 -4.0309e-01  1.1936e+01 -5.1900e+00  6.8269e+00  5.8526e+00
#>  -5.2317e+00  8.2800e+00  9.2374e+00 -1.3532e+01  1.0022e+01 -1.4072e+00
#>  -1.6480e+00 -4.9375e+00  1.0855e+01  2.9202e+00 -1.9057e+00 -6.5623e+00
#>   7.1908e+00  3.0071e+00  1.2720e+01 -4.1448e+00 -4.4275e+00  1.8524e+00
#>   6.8468e+00 -2.9011e+00 -4.6064e+00  2.9782e+00 -1.7799e+01  1.9231e+00
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,48} ]
```
