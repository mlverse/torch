# Conv_transpose1d

Conv_transpose1d

## Usage

``` r
torch_conv_transpose1d(
  input,
  weight,
  bias = list(),
  stride = 1L,
  padding = 0L,
  output_padding = 0L,
  groups = 1L,
  dilation = 1L
)
```

## Arguments

- input:

  input tensor of shape \\(\mbox{minibatch} , \mbox{in\\channels} ,
  iW)\\

- weight:

  filters of shape \\(\mbox{in\\channels} ,
  \frac{\mbox{out\\channels}}{\mbox{groups}} , kW)\\

- bias:

  optional bias of shape \\(\mbox{out\\channels})\\. Default: NULL

- stride:

  the stride of the convolving kernel. Can be a single number or a tuple
  `(sW,)`. Default: 1

- padding:

  `dilation * (kernel_size - 1) - padding` zero-padding will be added to
  both sides of each dimension in the input. Can be a single number or a
  tuple `(padW,)`. Default: 0

- output_padding:

  additional size added to one side of each dimension in the output
  shape. Can be a single number or a tuple `(out_padW)`. Default: 0

- groups:

  split input into groups, \\\mbox{in\\channels}\\ should be divisible
  by the number of groups. Default: 1

- dilation:

  the spacing between kernel elements. Can be a single number or a tuple
  `(dW,)`. Default: 1

## conv_transpose1d(input, weight, bias=NULL, stride=1, padding=0, output_padding=0, groups=1, dilation=1) -\> Tensor

Applies a 1D transposed convolution operator over an input signal
composed of several input planes, sometimes also called "deconvolution".

See
[`nn_conv_transpose1d()`](https://torch.mlverse.org/docs/dev/reference/nn_conv_transpose1d.md)
for details and output shape.

## Examples

``` r
if (torch_is_installed()) {

inputs = torch_randn(c(20, 16, 50))
weights = torch_randn(c(16, 33, 5))
nnf_conv_transpose1d(inputs, weights)
}
#> torch_tensor
#> (1,.,.) = 
#> Columns 1 to 8   1.8143   4.4585  -7.9210   4.8012   5.7963  -7.3841 -13.7057 -14.4838
#>    3.3155  -7.5558  16.1637 -12.0384   3.1279  15.8062 -19.0913  -6.7229
#>   -0.0095  -3.5173   5.2581  -7.3930  14.2839  -1.7473   6.4756  14.2400
#>    2.1695   5.6807   8.1891   2.0027   4.3754  14.5677  -1.6543  12.8926
#>    0.3893  -2.7946   4.6054  -6.4927   2.4885   3.4024  12.4936   2.4723
#>    4.4936   8.7134  -5.5254  -5.5835  -0.4480   7.6024 -14.1257 -13.4003
#>   -0.4366  -0.5704  -3.2567  -3.2212  -5.1598  12.1641 -15.9867  -4.2568
#>   -0.3542  10.6475  -0.1550  17.2083  -9.3312  23.9018   3.9687   5.5866
#>    1.7938   3.1120  -2.2602   6.1246  -8.8595   3.3421   9.0898 -14.9762
#>   -1.0431  -6.2460  -1.0992   3.2347  16.0008  -0.6231   2.3961  -0.9724
#>   -2.5036  -1.6233  -9.8053   3.1953  -0.6065  -8.3084  -0.3173  -0.4923
#>    0.2715  -3.1932  -8.6643  -2.9735   3.4325   4.9162  -7.2844 -20.5423
#>   -2.4553  -3.6580  -5.7217  -2.9959  -6.1367   2.8225  16.3570  -2.7585
#>   -0.3311   0.5858   8.7845 -12.7140   7.9186  -2.0969   2.6710   7.7518
#>   -1.2388   0.5740   3.8139 -20.0573  17.2250  -5.3844   9.0316 -13.0520
#>   -6.6792   2.2995  -8.0900  -1.3956   3.7782   4.2364   7.0295  17.8659
#>   -5.1352  -2.6419  -5.1916   0.8250  -7.3601  12.1743  13.3169   7.2807
#>   -2.7497   4.3888  -2.2114   4.5370   5.6021  -6.8048  -3.2586   9.1275
#>   -0.6794   4.9242   2.7830  -2.9482   7.9297  11.5992  -3.3721  -5.3572
#>   -0.2136   3.1888   3.6932  -3.1393   3.6029  -8.7517   3.2351   9.6702
#>    0.5004  -4.5786   5.4292   6.7049  -2.1959  -9.1048  -3.8082  -9.1023
#>   -0.8543   3.9921  -0.8550 -10.8001   4.8155   2.7585 -11.1825   7.9375
#>   -1.0847  -6.5881   6.5506   6.0645  -0.2384  18.6823   3.1830   4.4632
#>   -1.8486  -0.8692   0.1511  -2.2346  18.8581   2.1534   9.2836  -5.7370
#>    1.2709   3.3074  -3.6433 -18.8026  15.3486   0.2542   7.9208  -4.0677
#>   -0.5687  -4.4215   2.2900 -13.9114   5.2500  -9.2144   9.3359   3.9934
#>    0.8692  -1.0304   1.8761  -0.4801   4.7791   1.5649   1.0866  -4.6403
#>   -3.4141  -6.7191   7.3654 -22.0803  16.0671 -12.5741  -7.6989   1.7377
#>    1.4370   4.9838  -5.6053 -16.0584  -5.6433  10.2921   6.4453 -18.7667
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,54} ]
```
