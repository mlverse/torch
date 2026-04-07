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
#> Columns 1 to 8  -3.0173   0.7965   1.1748  -1.7441  -0.2370  10.3343 -14.3011  -9.7214
#>   -3.9823   2.7852  -6.0808  -6.1090   2.3258  -3.2432  -0.5854  25.4229
#>    3.9748  -3.4115  -1.5099 -15.8162  -3.1080  -5.5277  10.9365   2.5611
#>    0.1798  -6.5319   5.9119   4.6755  -3.8536  10.2246  -4.0890   3.7854
#>    0.1222  -6.4966   2.4282  -4.2191  25.1073  10.7234   4.5975   8.8824
#>    0.5596 -10.8971   0.6199  -9.4973   7.4486  -2.5853   9.5554   6.6996
#>   -3.8682  -1.8652   6.5447  -1.3319   8.3468   6.5494  14.6044  -7.2836
#>   -0.0753   1.0500  -1.4248   4.9759  -4.9223  14.7897  -0.3637 -16.3469
#>   -0.0806   3.4004   4.6866   3.9096  17.8548  -5.6388  -2.2356  -0.7226
#>   -4.3235   2.2207   0.7396  14.6499  -1.4412  16.0159  -2.6913   2.0870
#>   10.5879  -5.3600   1.6690  -3.2514   5.5229 -15.2530  -1.0972   2.2931
#>   -3.7098  -8.0132  -5.2265  -9.7759  -6.0728   3.1224  -6.7971   2.9657
#>   -7.7451   1.4220   0.8682   2.5141  15.9002  -9.3496   1.0353  -6.5644
#>    1.2430   1.0959   7.5344 -17.3918   5.0323  -2.7824  -1.1453  18.8218
#>    4.2534  -0.4096   8.6913  -5.9802   8.0728  -6.1307  -1.2654 -10.4989
#>   -1.8313  -5.3942  -6.5803   7.3869  -7.9288  -1.9017   6.9697  -7.9146
#>    5.7678  -8.2191   8.0966  10.7966   6.1024   8.1262  -1.0243  -6.2365
#>   -2.7602  -1.8966  -1.5225  -2.0961   2.0484   9.1302   0.9941  -4.0570
#>    0.3911   8.8926   4.4474   2.9504  -1.3264  -4.1755   0.4188  -8.4262
#>    1.7762  -2.4556  -0.0535  -8.4586   2.8303 -13.5981   4.7836  -6.5628
#>    7.9211   4.4327   8.1733   6.9849   4.0755  -2.6385   7.6579  -0.2110
#>   -2.7473  -7.4237   0.6368   6.7208   3.8218  12.6214  14.1889 -11.7566
#>    1.8260  -2.7674   8.2172  -2.9421  10.9617 -13.9830   0.5772 -16.8302
#>    2.9972  -0.6830  -5.6786  -9.4097   3.8777   0.9056   4.8993 -21.1029
#>   -0.6316   2.5966  -0.6568  13.6020  -8.3626   9.0971 -18.6042  -5.0268
#>   -4.2170  -3.3648  -5.5404   4.9461  10.2347   3.8952  10.5832 -11.4969
#>   -5.0874   4.7076  -4.5790  -3.4963  12.1506   4.1044  10.3716   6.8482
#>    1.1781   4.8575   5.4470  -3.5567 -19.2947   3.6593  -0.3538  -8.9751
#>   -6.9339   0.9938  -4.2693   4.0333  -8.5789   5.2887  -2.9813   0.7837
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,54} ]
```
