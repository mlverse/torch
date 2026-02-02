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
#>  Columns 1 to 8  -7.0144  -4.1277   7.0798   6.3364   8.7470   8.0297   4.5908  -3.7660
#>   -5.2860  10.5284  -2.5519   4.2223  -7.9589   2.5165   5.7519   2.7728
#>   -5.3414   1.9955  -2.7999  -1.5539   4.7893 -10.5710  -3.8897 -10.8862
#>   -2.3821   6.2552   6.6296  -7.1100   1.7924   5.1961  -9.5326 -13.5366
#>    7.2876   6.6276   3.6947  -2.7046   5.5710  -2.7902  -5.8053  -4.9035
#>    6.5158  -1.8778   0.6916  -7.4049   1.2305  -2.4279   2.2494  -2.9306
#>   -7.1591   2.5355  -1.2966  -6.7947   4.6222   5.1406  -5.7355 -10.4507
#>    8.2923  -2.5102   3.0247  -0.7884 -10.5194   4.1130  -9.3404   3.5228
#>   -9.0169   6.8308   1.1494  -5.4916  -8.4660  11.3356   6.0682   7.0305
#>  -12.2207  -7.7604   9.3588  -6.9974  -1.6502   6.9251  14.1754   8.2096
#>    1.9003  10.0118  -0.8861   5.7958  -2.3144 -18.7814 -17.7967 -12.2650
#>   -1.1274  -0.0180   0.8124   1.0571  -0.8163   1.1165  -2.4596  -7.2385
#>    4.6497  -0.1751   4.5253 -10.8381  13.0808  -2.0953  11.6059   7.7448
#>    2.5700  -2.2485  -3.5905  -1.0369   2.6314  -3.7983   8.5354   7.5580
#>   -0.5455 -13.6103   5.8355  -0.5726   4.1714   8.1067   0.5785  -0.4243
#>    8.3040   4.3271  -0.7334  -4.1864  11.3350 -17.5175  -3.0176   9.0982
#>   -0.3070   7.6589   1.4478  -3.4325   7.2494  -8.7283  -6.6121   3.6418
#>   16.2286   0.3319   0.0465   2.9444 -10.1858 -18.9261  -5.5365  -5.8939
#>   -1.5118   1.5468  -1.4936   0.9007  13.2183  -1.4004   7.9404  -0.1513
#>    3.0911   4.5358  -5.0088  -7.5713  12.0760   6.1476   4.4145   0.0564
#>   -3.8906   0.8433   6.4018   0.8777  -1.1889   3.0730  -7.6672  -0.0937
#>    8.0647  -0.6816  -4.0011  -0.0628  -1.8047 -10.2557  -2.8873  -3.4592
#>   -5.7131   3.7201   8.9534 -15.1189   6.8356   6.6407  -3.0120 -11.6921
#>   -1.1598  -8.0803   2.1300   1.1401  -1.9952   1.3270   5.6862  -2.1960
#>   11.6318   0.5379   8.8816 -10.3564  -2.2575 -18.6353  -0.6338   6.0717
#>    0.8569  -3.1052   0.0011  -0.7452 -12.7172  -4.0530 -14.0707   4.3690
#>   -5.4075   6.5183   0.0358  -3.6870   1.0571   1.1866   8.7882 -10.9919
#>   -8.9648   7.6337  -1.1119  -9.3109   2.9602  -1.1215  10.6024  -6.0274
#>    0.7193   2.7501   8.9377  11.6629  -5.6196  -6.5789  -2.0533  10.1106
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,48} ]
```
