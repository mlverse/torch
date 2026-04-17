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
#> Columns 1 to 8  -6.0267  -6.6903  -6.5087   1.5840  -4.2209   2.3045  -7.5299  13.9937
#>   -4.0625  -1.0026   0.8232   0.6968   7.1021  -8.0494   1.9803  -1.2644
#>    3.9017  -3.7278 -15.5464   2.9045  14.7552  -7.9823   2.7243  -6.4079
#>   -6.9310   1.8879 -16.4690  -3.9599   3.8752 -18.4247   8.7050  -5.0372
#>   -1.0585   5.4263  -1.7366  -5.7839  -3.7226  -3.0830   9.2210  -0.3262
#>   -0.9664 -22.1716   2.4796  -5.7297 -12.4890   8.7684 -15.6031   4.5701
#>    4.5607  -1.6407  -7.9708   5.8007   4.9049   4.0247  -1.2873  -4.2035
#>   -9.1919  -1.3128  -6.0654   3.5256   2.8506  10.2702  -5.0097   9.7427
#>   -4.4725  -8.3291 -14.2929   9.4520  -8.0734  -0.3276  -0.8000   0.4257
#>   -0.9110  -5.2555  -3.8827  -3.6472   4.1228  -9.4804  14.3681  -8.5832
#>   -4.7193   5.4829  -7.5747  -0.5506  -0.6764  -6.0117   2.0304   0.4475
#>    0.5762   7.5114  -6.1681  -5.4842   5.6663  -3.5138  10.5581   5.0962
#>   14.7602 -10.2245  -0.1754  10.4602  -0.5759  -0.3214  -6.7999  10.8483
#>    4.4212   0.9275   4.1103  -2.7499  12.1384   2.9211   0.6123 -12.7050
#>    2.9904  -0.5770   8.5900  -6.2570   8.4886  -3.3129  -1.0758 -12.4489
#>   -0.5538   6.5421  -4.1029  -7.6627   4.4318   1.9744  -0.0706  -9.6376
#>  -13.0972   2.2756   0.8144  -9.2790   1.9627  -1.4793   3.6947   1.9011
#>    5.9256   0.5995  -6.0564 -12.5344  11.9480  -1.5225  11.1816   4.9106
#>   -2.4641 -10.1014  -5.2139  -5.4007  -5.3693  -0.7461  -7.6675  10.0795
#>    1.5533  -0.8876   6.0335   3.2309 -14.2250   1.8036  -4.1143  -2.2790
#>   16.7952   5.7222   5.4787  -3.5500  -4.8011 -10.3744  -1.6429  -0.5689
#>   -3.7422 -10.2990  -6.5410  -7.9124   1.9967  -6.3033   1.4882  -1.2343
#>    5.5017  -3.8304 -10.0204   2.1859   3.8325   3.3464  -9.8294  -5.7217
#>   13.0281 -10.8814  -4.5866  -0.4243   6.7439  -2.7387  -5.0633  -4.9930
#>    1.4208   2.5341   2.6015   4.6000  -0.1273   0.2503  -4.8080   8.1257
#>   -6.2062  13.4110   5.0515   0.1404  -0.4657   2.6253  -3.9338   1.5806
#>    7.9734  -7.8364   8.0711   0.7148   5.2484  10.2942 -15.6487   2.0418
#>   -0.6790  -4.6456   1.1805  -0.8594  10.3538   2.6120  -5.1581   6.4042
#>  -10.3500   6.4517  10.8474  -2.7889   9.4012 -17.3867   7.5149 -15.7724
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,48} ]
```
