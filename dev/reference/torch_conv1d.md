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
#>  Columns 1 to 8  -2.1022  13.0821  -0.0970  -0.0279  -2.7126  15.9593   0.2661  -8.6402
#>   -4.7898   1.5844   4.2747 -12.2753  14.9653  -7.2284   1.3416   0.0031
#>   -5.1075   8.9753  -3.3924  13.7447 -17.4726  10.8937   5.3843  -0.0408
#>    3.1677  12.7939  -4.9163   0.4240   5.4813   4.0590   0.5660  -1.0535
#>   -3.5267  13.6115   9.5989   9.0244  -0.3730  -5.1702  -4.8628  -3.9419
#>   -6.1518 -19.3415  -2.3094  -4.3521 -16.6526  -3.7222   0.4737   8.0007
#>  -13.3207  -1.1880  -0.5117  -6.3022  -1.2788  12.5334  -2.0678  -1.9533
#>   -2.5643   9.7573 -10.7282  11.8455  -2.6905   2.4362  -4.4904 -19.8238
#>   -5.9503 -11.2132  13.0632 -13.1465   5.4073  -8.4374   0.6431   9.8528
#>   -2.3343  -3.7306   6.6923  -6.7151  -6.6659  -6.2879  -5.0405  -7.8015
#>    7.5221   5.3462   5.2386   9.5921   6.2198   4.9804   4.7617  -7.2045
#>   -5.7773  -3.6255   4.7621  -3.4394   2.4578  -9.4284   0.3196   7.2164
#>    2.9373   8.9738  12.2974  -1.6174  -3.1690   7.0896  -8.7889   8.2969
#>    8.4417   6.5867   7.5603  -3.9387  -5.8253   3.7991  -1.1375 -11.3179
#>    1.4303 -12.3277   2.0530  -8.5633   8.9847  -5.4711   4.3237   7.7792
#>   -3.5075   4.5679   0.1768  13.9197  -8.9645  10.4106   6.7535   7.8268
#>    7.9418   7.4272 -16.7256 -12.9152  15.8204  -3.3158   2.7755   4.0029
#>    4.7606  -1.8340  -1.2287   0.0382   5.6898   0.7328   5.2380   6.5299
#>    0.9346  -5.8737  -4.7033  -5.0583 -12.4485   1.1589   1.3538  -2.7114
#>   -5.4260   8.9529   4.3612   4.0386  -5.3060   5.3189 -14.8828   6.3743
#>   -8.6093  11.1289  -8.3914   0.0551   6.2728   0.0167   1.7325 -18.9212
#>    5.4750  -5.0999   6.4440 -12.3232  -3.0220  -4.8097  -0.8319   7.5017
#>   -3.4314  11.2534  -7.5667  20.2366 -13.8226  15.2641   3.7415  -6.6624
#>   -1.9060  -6.6683  -4.6519   1.3146  -9.8588  -1.6451 -11.9118  12.8671
#>   -4.9107  -1.9121   6.3639  -8.4958  10.1525  -0.2721   0.4320  12.0062
#>    6.4799  -5.6075  -0.2851  -3.2043  -2.6249   7.2846  -5.1582  -6.6010
#>    1.7996  -1.1307  16.8877   0.9750 -15.6109   6.2788  -6.2962   5.9229
#>    1.7472  -4.6471   5.2806  -6.2741  -0.5619   4.8003  -1.7848 -10.5011
#>   -4.8881  -4.0574 -11.7585  -2.3359   9.1189 -19.2972  -4.4632   9.2826
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,48} ]
```
