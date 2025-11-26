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
#>  Columns 1 to 8  -3.2573   2.9532   4.0885   1.4055  13.6152  -9.6572   3.5430  -2.0005
#>    0.0667  -2.7445  -4.7379  -5.3522  -8.1816  -1.0104   5.8767   7.9252
#>    3.1566   4.1248   1.6749   9.7312   7.4270  12.4900  -7.0353   2.1060
#>    5.0717 -10.6764   0.2697 -15.9251   5.4533   1.9363  -2.1118   0.7160
#>   -7.2159  -5.2244   4.3264  -3.6072  -1.6993  -1.1325   5.3964  -0.4847
#>    0.0884   0.1804  -3.0014   1.2705   7.5776  -3.3853   0.7669  -9.7782
#>   -1.1800  -5.0945  -0.4773  -2.6290  -6.4455  -4.0715  -3.8701  14.0288
#>   -2.1887   3.7885   6.6559   9.1927   0.7615   0.7292   5.7062  -1.1481
#>    2.0020  -1.9780   8.4716  -2.5897   4.5035   9.7599   3.3355   8.6611
#>   -8.2570   3.0853 -10.0956 -10.0834   5.6301  -4.0419   7.5967  -8.1802
#>    4.9178   6.7718  19.1764   3.6912   0.4916  -8.8475 -11.3142   4.0928
#>    1.6343   0.9909   2.0279  -5.0318  -4.8369  -2.1933  -8.4173 -10.4756
#>   -0.6881  -0.7128   3.7963  -2.5602   2.7677 -11.0210  -1.6772  -4.0715
#>   -9.9718   4.7804  -1.2903   5.3721   6.7404   8.2824   3.5543  -3.5140
#>    1.3932  -0.2000   4.1953   3.6080  -4.8945  -3.5581 -15.7891  -2.6137
#>    5.7781   1.3667   4.5577  -9.0260   0.5114   8.2136  -3.4210   4.9530
#>   -7.3268 -11.1374  -1.9996   4.4727   7.9787  10.6809   6.6179  -4.2382
#>   -1.4383  -7.3204   5.9341  -2.7202   3.0322  -0.3241  -4.9028   5.9438
#>   -9.3440  -5.0710  -4.7740   3.1878   2.7826  -5.3432 -10.9059  -4.7825
#>   -3.0980   6.1100  -6.8548  -0.6470  -5.3560  -1.4366  -0.0833   9.4530
#>   -1.5158   0.8899  -2.6888  -0.2415  16.9345   2.1567   7.4966  -4.8864
#>   -8.0359  -4.2464  -1.7751   6.9952   6.5229   6.7939   1.3401   3.4679
#>    2.9009   1.8637   8.1524  -0.0757  -1.6167  -7.5773  -4.2819  -4.7347
#>    2.5318  -4.9670 -10.2921   0.2788  -9.8839   6.7350  -1.8798   1.1626
#>   -4.6899  -9.3729  -5.0958  -8.2870   0.4033   0.1089  -3.3686   2.9812
#>   -1.2953 -10.0803  12.4854  12.9660   6.1420   7.8938  -3.8455  12.4042
#>   -1.4124  -7.6533  -0.4921  -8.8346  -6.8546  -4.8491 -14.8609   4.8746
#>   -1.8734  -2.1042   5.8541  -1.5907  -0.8522  -2.8019  -6.4742   4.5182
#>   -2.5429  -2.6185   5.3455  -7.8497  12.9223  -1.4552   3.7631  -3.5912
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,48} ]
```
