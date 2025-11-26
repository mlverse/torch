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
#>  Columns 1 to 8   2.2034   0.3855  -3.2783  -4.1116  -2.3115  -0.1433   0.6804  -8.8403
#>   -3.7658  -7.7352  -8.7379  -7.8905  -2.0115   3.3701   4.9871 -16.5174
#>    7.8215  -7.4255  -2.1309  -9.8091   6.9636  -2.4814  -2.1258 -11.3778
#>   -1.9711 -11.1179   1.3634   6.9388   6.5471   8.2038  15.7385  12.3004
#>    3.3127  -3.3723  -7.8513  -2.6552  -5.6973   0.0632  -1.5263   5.5565
#>   -1.8083  -4.1581   4.4666  -9.1270   5.7641  -6.9707  -7.8821  -1.1317
#>    0.2847 -11.6221 -13.1246  -4.9528  13.2055  -8.6927  -9.9201  -1.0095
#>    3.6622  -3.3339   4.6253  -5.7997  -5.5728   9.2731   6.5970  -7.8079
#>   -5.1031   1.3851  -6.0737  -2.6855 -18.2098   6.3914 -11.6444   4.6710
#>    1.7236   5.1078  12.7150  -4.3106   7.1048  -9.2683  16.0045 -22.5709
#>    1.3624   5.5577  -8.2996  12.1082   2.1409   6.2748   4.0433   7.8685
#>   -0.4464   3.6973  -1.3408  12.7351   1.3505   1.9401  16.3705   3.4130
#>   -0.5676   0.5441   3.3293   3.0280  -8.3091  -8.2024   8.4852  19.1686
#>   -3.0807  -6.6292  -3.6333 -11.2908  12.8541  -3.8834   4.6539  -7.4118
#>    1.5621  -2.4266  -7.4805   5.5534  -0.9055  -8.7007   9.0539   9.1565
#>   -6.7700  10.0147   8.0652  13.4742  -1.1687  21.6439   2.7316  -3.8657
#>    2.3863  -8.7283  -3.5490 -17.3009   4.6880 -12.2192   4.5610  -7.8730
#>   -2.2231  -1.4978  -0.8522   6.1933  -3.3933   3.2760 -13.4216   0.2997
#>    4.4688   4.9699   6.5143   1.5080  26.3757   6.6273   9.1061  17.9299
#>   -8.6143 -11.1067  -4.2538  -4.7305  -6.5050  -4.7532   6.2200  -8.7675
#>   -0.7498  -6.9793   9.7560   6.3303  -0.6213   9.0262   1.6254  -0.9358
#>    4.4271  -2.6184   5.3785  12.3834   9.4347  -3.8116   7.2821   8.0813
#>    5.9495 -10.2365  -9.3556   3.7452   0.6123   2.4425   1.7154  11.7910
#>    3.3842   7.9125  11.4428   1.5760   4.0971   7.8377  -2.1764  -7.3660
#>   -2.4493   7.0394  -1.1832   8.7881  -0.1850 -12.2491   0.8468 -12.4982
#>   -1.9841  -7.5946 -11.0456   4.0766  -9.3997 -14.7509   0.8228  -8.5361
#>   -1.2261   5.7867   2.7455   0.3012   1.3693  -1.2213  -6.6094   4.6115
#>   -3.2598  -4.3930   0.0835  -3.5600 -11.9035  -4.3179   5.4350  -5.7236
#>   -2.7380   0.7348   9.6475  -2.0549  15.5819   3.6011  -6.9350  -4.7819
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,54} ]
```
