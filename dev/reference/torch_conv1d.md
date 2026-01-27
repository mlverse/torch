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
#>  Columns 1 to 8   7.7821   1.1401   7.0596 -18.0148  -8.0119   3.5312   6.3678   1.1577
#>    1.4168 -21.7658  -3.8629   3.7836  -2.8730   3.3767   1.3107  -7.6912
#>   -1.2900   1.3924   0.4293 -12.1456  -5.1198  11.9490  -7.0972   0.1647
#>   -3.9984   5.9896  -9.3157  -5.7545   0.2538   4.3508   3.8594   2.2356
#>   -1.3127  19.0513 -11.2265  -3.7089  -4.9470  -2.9273   5.2524  -2.3379
#>    6.9037   8.4869  -0.1168 -14.1164   7.8168   2.3922  -6.4525   6.7913
#>    3.2033  14.0674   7.1300  -1.6438  -3.9614  -2.0177  -5.9154  -3.2822
#>   -3.8510   8.7331   4.5380  11.4523  -0.8700   6.0459   7.0312 -12.1747
#>   -2.8827   9.2593   8.2983  -0.8874   3.0346  -5.1481  -4.5896  -0.4153
#>   -0.0607  -0.8083  -0.9835  -0.0790   5.9826   2.1725  -3.8092  11.2017
#>   -7.0652  15.5899  -1.6786 -13.9086  12.0909  -1.3276 -20.9890  13.7807
#>    0.4774  -0.0128  -0.2353  12.5776   1.0890  -6.0474  13.1473   4.1201
#>   10.3552   5.7057 -11.3293  -2.8307   1.6694 -11.5058  -5.5629   6.6374
#>    7.4876  -1.3666   6.0264   1.9139  -2.6408   6.2576   2.6033  -6.8402
#>   -5.1727   1.0549   9.9360   1.8297  -0.1804  -2.1706  -6.6508   8.5265
#>   -2.4241  -4.2964   2.4751   1.0475  -1.7561  -7.6616   3.0063  15.1142
#>   -3.6064 -13.1878  -5.6382   5.4519   8.3442  -0.9002   9.3149  -4.8872
#>    1.6292   5.3288   4.2803   8.9457  -5.8876   4.4505   0.5132  -4.3812
#>    1.6326   7.3562  -0.3425  -9.7534  -0.2727   3.4866  -3.4696   7.8609
#>   -6.6557  10.4909   2.9039   1.0667  -1.8801  -1.2761  -2.9188 -11.0050
#>    6.2751  -1.9554   9.2415   1.5443  -5.9285   2.2983   1.0393  -0.3675
#>    0.5540  -8.0793  -4.0453  15.4465  -5.4665   8.5274  -9.2435  -5.6795
#>    0.3882   6.7377   9.4104  -1.1289   7.6823  13.5474  -1.6195   8.9032
#>   -7.8136   5.9994   1.1162   7.6274  22.1626   6.3166   6.9810   5.6684
#>   -0.7276   0.7945   8.8184 -15.3950  -7.8922   3.8768  -6.6414   0.3786
#>    1.8089  -7.1999  -2.9335   4.4606   0.1712   0.9870  -3.9326  -5.8008
#>   -7.6991   5.6514  -6.8531   5.1307   9.7328  -1.1070   4.3148   2.8988
#>   -2.4336 -11.0794   5.7383  11.9821  -2.5089  10.2715   5.2683 -15.4027
#>   -3.6935  -7.8457   5.9850   3.0752  -7.3826  -2.2674   3.2610  -2.7119
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,48} ]
```
