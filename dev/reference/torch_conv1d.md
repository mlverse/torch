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
#> Columns 1 to 8   8.6297   4.8263   2.7139  -1.8182  -4.9107  -2.1260   0.0531  11.9104
#>   -7.6191   2.5912  -5.9308  -6.4301  -8.6131  -8.2743   0.9462  -1.3322
#>   -3.2111   9.1860   4.2890  -4.8730   9.1951  -4.3248   7.5913   0.1690
#>   -5.5712   1.1567   2.2186  -2.0030  -7.9575   6.7057  -1.9539  -4.0466
#>    0.4099  -6.3974   5.1544  -2.2151   1.1463  -8.3512  -9.2182   1.9351
#>    1.1108  -1.5058  -7.0625  -5.1365   5.1963   3.8669  14.1480  -6.1982
#>    7.4318   9.8204   4.5235   1.5133  -7.4921  -4.1718   2.5402   4.7246
#>    2.8930  11.6633  -7.0685  -6.0611   4.2881  -2.4021  -2.2939  -5.2583
#>    9.5094   7.5196   7.9952   5.0807  -6.8165   3.6799  15.0085  -7.6699
#>   -5.6988   4.0486   0.5191  -0.2423  -4.8427 -16.9643  -7.3242  -3.6550
#>   -0.9177   2.1060  -1.8950   1.4522   8.4210   5.7554  18.1320   2.1003
#>   -1.3339   7.9689  -2.8041  -5.3220   5.6304   6.6063  -5.2814   9.3930
#>  -10.0690   3.1302  -1.4571   2.9797  -0.8248  -3.4994   9.4969  -6.7218
#>   -0.7424  -8.9488   2.6737   0.0894   2.2789  -5.0495 -11.6957   1.1769
#>   -3.2797   0.3199   8.5508   2.5641  -6.9351   6.1256  -0.2607  13.3857
#>    4.2146   1.8293  -3.7035   0.0552  -0.3457   5.7362  -1.4391  -0.6040
#>    1.9150   4.5508  -2.3577   8.8002  -2.0786   1.2887  12.2871  -2.8474
#>    3.1175  -9.4547  -5.3396  -1.5128  -7.4906   5.4094   7.1580  -0.1142
#>   -1.4978   4.3097   9.6367  -4.1225   0.4569  -3.0673  -4.2527   6.5492
#>    3.0946  -5.9995  -4.0387  16.5588   4.7013  -1.1555   5.9218  -6.0571
#>   -5.2358   3.0287   2.0012   3.8816   3.3400  -1.4829  -0.8582   3.0452
#>   -1.6689   6.2721   3.8742  -1.8976   6.8562  -9.4392  -1.3455   4.6777
#>    8.0178  -4.9020   5.5176   7.4587  -0.7038  -9.2408  -1.5242   1.3984
#>    3.9686   6.4665  -3.3195  -4.6166  -0.1602   6.3445  -2.8320   0.8690
#>   -5.2853  -9.0636  -4.0429   5.5771  -1.2980   6.7518  15.3554  -3.1537
#>   -1.1843  -1.5393   2.7448   5.5057  10.9106   0.8681  -0.6613  -4.1220
#>    9.9510   0.9254   3.2389  -2.3205   0.6834  -1.9663   4.7849   2.2327
#>    1.3679   4.3160   3.6014   1.9505   5.4584  -2.0593  -4.3534   8.4652
#>   -1.7961  -3.5179   4.5093   3.7006   2.5679  -2.8633   3.4665   5.5932
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,48} ]
```
