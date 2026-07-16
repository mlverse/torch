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
#> Columns 1 to 8  -1.9298   6.3626   4.2852  -3.9345   5.3961  -6.5190   5.2129   4.6899
#>    0.8748 -13.0245   0.4165  -9.4052  -0.0934  12.5683 -12.2637   3.1642
#>   -4.6507  -1.3114  -8.1886  -1.7645 -18.1542  -2.0604   6.8026  -0.2677
#>    5.6317  -7.1805  -3.9147   8.9821  -5.1317 -12.1343   1.1120   5.7826
#>   -7.2357  -0.3645   9.0885   4.3796  -2.7078   5.5609   2.9083   8.8643
#>   -9.5015  -1.8596  -8.6801   5.1548   3.7781   4.8615  -6.8111  -6.0667
#>    3.8719  -5.6678   2.9779  -0.4287  -2.4228  -0.8244   0.2029  14.5869
#>    2.7688 -12.0265  -2.3840  11.9211  13.0246   3.0742   2.4634  -3.3607
#>    5.2317   1.3063   4.6631   2.4199   5.0991  -3.2300  -4.5769   0.1054
#>   -3.7690  10.8577  -5.1228  -7.3378 -21.2856   7.5357   1.7712   8.6328
#>    4.6354  10.8211  -3.2434 -11.2625  -6.3087  -0.8301  -4.4142  24.6187
#>    8.7892   1.3868  -9.3726   8.6565  -5.0501  19.4789 -17.2647  -0.2875
#>    5.8607  -6.9494  15.5504  -3.7657 -17.4631  11.1783  -5.5948  -0.5723
#>   -6.7000  -7.2137   6.9448  -3.2434  -3.8582   8.6985   8.7108   8.8012
#>    6.2204  19.6669  -2.4469  -6.0453   5.0391   4.5777   0.5885  -1.0951
#>  -26.7209 -10.0441   6.4832   0.5437 -12.9525   1.5800 -18.4743   2.7642
#>   -1.1524   2.4986   8.1982   2.3516  -4.8492  -6.5285   2.8132  -0.3269
#>    7.6819  -1.5314   1.6344  -2.5514   7.3236   7.6676  -2.8242  -1.8371
#>    8.2966   5.5824   3.9846  -0.0070   6.7412   7.0075   2.7171  -9.8267
#>    1.5018  -2.3297   6.4235  -7.0243 -10.0396   0.5277   0.1193   5.4896
#>   -3.2951   4.9652  -0.4867  -0.8696   5.1728  -3.6390   0.2141  -6.7439
#>    7.2861   7.7472  -7.6320  -4.7148   9.5975  -0.5938  -6.4665   3.0432
#>   -3.1042 -12.2073   5.6291   2.9584   7.9102   0.7731   6.6514  -1.6835
#>   -4.8511 -10.0848  -7.3932   2.4102  -2.0476 -10.8969   3.4488   2.6129
#>   -9.5503  10.2941   6.6023   5.0600   4.0310   8.1511   2.0660 -15.2098
#>   -5.4945   7.6590  11.7106  -3.6691  -0.5730  -7.1847   2.7146   7.3288
#>   15.5680  -1.6918   0.0264  -6.8391   5.5766  -1.0386  -4.1338  -0.7349
#>   -1.4910 -10.2552  13.2071   1.2411   3.6796  -0.1976 -11.8439  11.6231
#>    6.7501  -7.8906  11.8479  -2.7976   3.8064 -15.6502  -4.3122   6.9093
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,48} ]
```
