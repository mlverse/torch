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
#> Columns 1 to 8  -4.5773   2.6147 -14.3122  -2.1602   4.0987   8.0053  -8.5743  -9.2333
#>    9.3975  -4.6597 -11.1311  -1.9025  -1.2187  12.5010   0.9536  18.7899
#>    5.4156  -6.7443  10.5718  -6.3605   3.0685  -2.9289 -11.7281  -9.9288
#>   -5.5331   3.0013  16.6530   7.6428   2.2904  -3.6188  -2.1749  -8.4055
#>    4.4273 -13.0618  15.7483  -9.5707  -4.5583  -9.6225   3.8433   1.9245
#>    5.8947  -7.2893  11.1714   3.1198  -5.9954  -9.8512  -7.9350   0.0940
#>    2.0844  -2.7398  -5.4162   3.3635  -0.7274   0.5672  -2.1196   5.4851
#>   -2.0960  -1.2156  -4.2970   0.2308  -2.5830   3.6801  -8.2397 -19.3972
#>    3.3475   7.2797  -3.1015   4.2243   1.9868  -0.6200  -0.2765   3.4347
#>    0.8350  -9.3403  -2.0197   3.4559  -1.1479   0.3846   2.9426  12.6746
#>  -15.1596   6.1491  -4.3970  -7.7683   3.2763  -1.4938   8.6902  10.6794
#>    0.0566   1.5774   5.0116   2.7400  -8.4679   9.5227   4.1253  10.8683
#>  -13.6310  12.7769   0.4181 -22.4119  10.5037   6.7665   1.8812  -8.4994
#>   -7.1102   3.0618  -6.4561  -5.2512   4.7363   9.8656  11.5740  -4.2634
#>    1.1386   1.1563   5.9478 -11.4225   1.3340   0.8291  -7.6447 -13.6351
#>    2.3424 -10.8518   2.2517  10.2346  -5.2450  -0.2447  -2.5452  10.8779
#>   -5.5373   4.3440  -7.4951  -7.4895  -9.4776   9.6069  -0.6105 -19.5954
#>   -3.4123  -4.2435   4.7803  -6.4821   6.6088   8.1498  10.0620  -0.1913
#>    0.2724  -3.9494   4.8659  11.5953  -9.4398  -6.0080 -12.1608  -9.5690
#>    4.6175  -8.2522  -3.2089  16.6523  -7.8521  -2.8815 -19.0868  -6.5054
#>  -10.4280  10.6491  -6.7516   4.7841  -1.1442   1.5845  12.3176   4.9313
#>    5.7118   1.5266   6.0667  14.1252 -15.9411  -3.7228  -2.1659 -10.4912
#>   -6.9439  -4.1180  -0.5075  -4.4187  -2.6995 -16.8056  -7.0258   3.0761
#>    5.1417   4.5260 -10.1249  -2.8191  -3.3379  -0.2802  -4.5738  -7.8067
#>    8.5716  -6.5256   3.6289   7.4135  -7.5905   8.8333   4.9925  -0.8931
#>    4.2394   0.2720  -0.5692   4.1494  -0.4732  -3.9878  -0.1256   4.6988
#>   -8.4185   5.1636  10.7442   4.8214  -9.4389  -1.0210  -3.7229   0.2250
#>    1.7873  10.3047  -2.0821   3.0603  -8.5218   1.6490   3.0864   6.9026
#>   16.0665  -7.8096  -5.6823  -2.8957   1.0273  -0.0063  -2.8696  -1.3478
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,48} ]
```
