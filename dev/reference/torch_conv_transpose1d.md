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
#> Columns 1 to 8  -1.4666  -1.3563  -9.2965   3.4849  -4.5419  -2.5685  -7.8769   1.1146
#>    1.5312  -8.4130  10.6301   5.1325 -12.6676   2.3120  -6.2895 -15.3605
#>    3.6695   0.1646  12.2720   7.0529  -8.0879   4.5391  -2.4596   2.6828
#>   -3.3698 -11.5070 -14.6862 -12.2128  13.1294  -1.1267  -2.4963  10.1924
#>    4.9929  -7.0658   5.1942 -10.5299  -7.9816  16.2860 -17.5972 -10.9302
#>    2.7929   7.8138  -5.0867 -18.2886  -2.0407  -9.1444 -15.1068  -4.6222
#>    1.8677  -2.5466  -8.6449   3.0749  13.2507   2.6488   2.8783  13.1020
#>    1.5469  -5.4262  -3.1102  12.1243  -0.2669 -20.0522  -0.1903 -11.2255
#>   -0.2480  -4.2463   1.1737  11.3429  -4.4074   7.5265   3.3312 -14.5456
#>   -2.3027  -8.8151   7.9599  -0.5320   0.6844  -2.6234  -4.3312   4.5173
#>   -5.4767  -1.2360  -1.4923   7.0261   1.2711  -7.2607   0.0365   2.3229
#>   -0.3155  -5.7710   4.6669  -0.1452   2.5654   3.8453 -11.3158  -0.0820
#>   -5.7141  -3.9640   3.6241   8.0447  14.6430 -11.6491   5.4307  -7.2726
#>    2.5097  -2.6537  -2.6370   4.3747  -9.2024  -4.8356   5.3316  -6.6513
#>   -3.1968   3.6371   1.8246   0.8143   6.1359  -0.4210   3.6116  11.6458
#>    6.2377   0.3576  -0.2387  -2.5876   7.9248   7.5618  -6.3363   0.7582
#>   -9.1463  -0.8992   1.2534   9.6682  -9.4387  -2.3557  19.3195  -1.7516
#>    1.0160   1.3686  -0.8986  -4.9835   0.1812   5.9553  -1.0935   7.1718
#>    0.3865   2.6660  -2.4905  -2.8462  -0.7606   2.9661  -5.6069   7.0243
#>    1.2917  -6.2852  -3.1225  13.7702  -7.0799  -2.9002   4.7043  -5.2138
#>   -1.1746  -0.7449  -1.0268  -4.6795   6.0414 -14.2019  11.2037  11.4254
#>   -1.5996   7.9917   0.7976  -5.8179  -4.3931   4.7045   2.6353   6.0755
#>   -1.8459 -13.9613  -3.1368  -2.8800  -3.0698   8.9247  -8.3872   6.5089
#>    0.4600   1.8557   9.0410  -0.8445   1.2339   2.5174  -4.2210   2.9154
#>   -6.4964   0.0532  12.8831   2.4796   3.2428  -8.5869   8.0578   3.3183
#>   -3.9095  -0.5895  12.2917   7.2205  -8.9735   2.6864  -5.4881   7.7456
#>   -6.3000   3.4059   7.1822   6.1271 -17.1961   2.4002   4.0412 -19.2023
#>   -3.9083  -3.5045   5.7004  -2.4587  -1.3451   9.4032  10.3354  -4.6971
#>   -3.5809  -6.6281  -5.5389  -6.1722  -2.7409  -0.5687   6.7904  -0.9895
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,54} ]
```
