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
#> Columns 1 to 8   1.4615   1.2908   1.2671  -6.9663 -10.3063  -6.8823   0.3302 -10.2312
#>    8.6450  -0.4085  -4.4661   9.7396   0.1261  -2.4617   8.2007  -6.3515
#>    4.0756  -1.8183 -10.4743   6.1236  -0.0288   4.6126  -1.4320  -6.4714
#>  -10.7878  -2.5653  13.8188  -5.4470   4.6366  16.8054  -0.0746   2.2504
#>   -0.9202  -5.8492  -4.4256   1.3876  11.4231   4.2360  -4.3945   6.1901
#>    4.7834  -1.3764   3.0160 -10.0039  -9.5197   5.9873  -7.2847  -1.2113
#>    3.0568  -3.9174  -5.5149  -3.2682 -12.7448  -9.4994   9.9715 -12.1403
#>    1.8685  -6.6764   3.6233   0.4773   8.4131   3.9630   1.6251  12.2925
#>    7.3256   4.4822  -3.3550 -12.6656   4.5432   5.3761   0.7867   1.4555
#>    3.7951  -5.6825   7.2436  -5.6577   4.8548  -6.7410   0.7537   2.1238
#>    3.2107   1.4452   1.5684  -3.0141  -0.7565  -3.1253   8.2315  -9.5172
#>   -3.6830   5.3323  -0.9594   6.3408   0.9637   5.5589   0.1684   3.6134
#>  -10.2543  10.4039  -7.9820  -5.1548   0.0027  -2.9226   2.7252   4.1407
#>    8.0680   5.3910   1.9570 -14.1544   8.9861  -7.2515  -3.7637   1.2927
#>   -6.5752  12.6568  -2.9285   6.1341  -7.3680  -2.8949  13.6012  -1.0185
#>   -2.0028   1.5324   1.9631   2.7631   1.2863  -6.5545   7.2734  -3.0657
#>   -3.9287  -5.9420   2.3490  -4.2463  -0.4491   2.0105 -11.1312   3.5421
#>    8.1757 -15.7401   3.9733   8.1982 -12.3643  -3.4431  -2.8671  14.5056
#>    1.2173  -8.5129  -3.8186   7.0474   6.0148   0.0917   1.3227   0.1373
#>    2.4177   8.3673   2.4621  -4.2291 -12.1173   1.5984   0.3174   0.3634
#>    2.6424  -1.4525 -10.4005   3.2584  -5.5645  -0.6658  -5.2759   3.5764
#>   -1.4612   7.6479  -7.5330   7.8278  -4.9852  -4.2933   3.3133 -10.6169
#>    1.5235  -5.3261   6.5337  -1.7760   2.3870  -3.0706  -3.0927  -0.8813
#>   14.5863  -4.6933   2.0494   7.2487   1.0536  -7.2452   1.5297   2.8349
#>   -4.2939   8.8645  -9.2766   4.7927  -2.2447  -8.4683   1.2531  -0.1184
#>   -2.7342  -5.2044  -2.5007   1.5080   5.3349 -10.3502  13.8029  -1.8147
#>   -4.9276   7.8918  -1.9237  -8.9261  -3.1200  -2.6340 -12.4504   4.5008
#>    5.6338  -1.5104   1.8324   2.8508  -2.9890 -11.5600  -2.6905  -3.8738
#>   -4.6700  -0.3331  -3.4370   1.4978   7.8623  -5.6665  -2.0595  -2.6367
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,48} ]
```
