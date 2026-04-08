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
#> Columns 1 to 8  -1.0785  -8.1779  -2.6281 -10.7019  -0.2121   6.6683 -11.4541  -5.3495
#>    2.5638   3.7861   0.7261   3.8185   5.0835   0.5959   1.2259   8.2509
#>    2.0329   7.4088  -0.4756  -6.6850  -1.8198  19.0322   2.9743  -6.3536
#>   -2.0140  -9.1056   7.1315  -0.7646   2.7548  -4.3280   3.5369  -3.8685
#>    0.5232  -7.7771  -1.3467   3.8147   4.4776  -9.2533  -8.6868   4.0153
#>   -0.6408  -8.1453   2.4174  -6.5502 -11.2921  -6.3900   4.7341 -12.4723
#>    3.0839   1.8356   2.0872   9.4359  -0.7821  -4.2376  -4.5362 -14.4945
#>   -5.4428  -4.1766   5.2526  12.2790  13.5962  -1.1982   3.3952 -13.0181
#>   -1.3628  -8.8610  -9.1728   0.0306   1.2201  15.4645   0.2856   5.8503
#>    3.2025   5.0218   5.8664  -7.3339   4.1011 -18.1270 -19.7349  -6.0717
#>    0.0304  -2.1695  -0.2140 -13.6483  -9.7008  -9.2381  -1.9352   1.1776
#>   -2.0350   2.0447  -2.0668  13.4732   3.2299 -11.8686   2.4122  -8.3610
#>    1.3654 -10.0851 -15.4365   5.0141   4.8371  -0.6990  -4.0001   2.2255
#>   -5.7650  -1.5984 -10.4633   7.7689  -3.5806  -6.9817  -5.5050  12.3190
#>    1.1518  -5.5551  -0.1175  -2.4404  -3.5777 -12.8203   4.4057  17.0090
#>    1.0554 -10.2044  15.3389  -0.3597   3.2284  -3.7974   2.8500  -0.5931
#>   -2.8804   8.4753  -8.4942  21.3841  24.0406   3.4218   3.4632  -0.3991
#>    4.4663  -7.8500   6.8118  13.4003  -5.7864  -5.9575   1.5297  -3.4587
#>    2.5756   5.6817  -6.1450   0.2645  -8.0020  -8.3788   9.7764  -2.3511
#>   -0.1842   0.8923  -3.5561   0.3110  -0.2299  -3.1698   6.2559   0.4874
#>   -7.8615   2.1006  10.3615  11.1625  17.4224  17.3716   5.9429  -4.7401
#>   -0.4869   1.8438   4.0065  -7.8009   1.1913   8.0978  18.7649   3.2845
#>   -3.1243   0.2896  -1.1747   5.1757  -8.7206 -20.7463 -10.7113  -2.7157
#>   -4.0092  -1.6458  14.3663   3.2888   2.0207  -3.0488  19.5105  10.7763
#>   -4.7482   4.7360  -1.0765   6.4146  -2.2579 -13.2147   4.3716  -2.1084
#>    0.3075  -6.4900   0.4602   3.9060   5.7812   8.9152  -6.7424 -17.5726
#>   -0.0783   2.1561   0.5314 -12.6729 -14.6716 -17.8604  -9.8576  11.5804
#>    6.6348   0.8384   1.9875 -14.0564   1.4658   0.7725   7.7731   7.2426
#>    1.1305  -6.4405   9.0195  -5.7282  23.6339   7.7103  -4.7822 -11.3825
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,54} ]
```
