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
#> Columns 1 to 8   0.1732  -9.1792  -6.0508   0.9626   9.4998  -9.8652   3.9236  -6.5179
#>    2.4293  -6.2329   1.4427   0.8655  -1.1172 -14.6097  -8.2885   2.2607
#>   -6.2029   1.6667   0.1525  -4.0411  -5.7364   9.1180  -8.0410   6.7609
#>   -0.1943   6.4600   1.1415   1.2680  -2.6796  15.9691  -8.6830   7.2652
#>   10.6447   4.7201  -4.9805  -0.0910   9.6915 -14.0591  -6.9948  -0.5182
#>    8.7734   6.3083   4.9338   4.1991   8.7229   3.5382  11.1553   3.7615
#>    4.5994 -14.0097  -5.4886   2.0583   1.7510  -4.6432   4.2671  -2.6466
#>    4.5702   9.5522   1.2533  -0.5480  -0.1053  11.7940  -7.6269  -1.8257
#>    2.0361  -6.2074   3.9804   1.8150  -2.0928  -6.5001   6.0019  -8.4032
#>    2.3759   1.3640   5.0110   0.5394   9.0379   0.6327 -11.5885   7.4573
#>   -5.2253  -0.9090   4.0920   1.9245   1.6809 -10.8261   9.5025   7.9970
#>   -9.1237  10.6417   8.6887  -7.0363  -9.6616   9.6547 -12.6597  14.9800
#>    1.6060  -5.6128  10.0261   2.7464  -0.4203   7.7072  10.2490 -12.2586
#>    9.3918  -1.2863   7.2845   2.6528   7.6655   3.3494   4.5300   2.5917
#>    7.4941   0.2014  -4.6070   3.1734   9.1331  -7.9372   1.4830  -6.9954
#>    1.5603  -7.2621   0.4867  -1.7704  -6.7390   2.1587  -2.7499  -3.5179
#>   -4.4972   4.7264   1.9513  -4.5109   8.7286  -6.9411  10.2274   5.4821
#>    2.3824  -0.0383  -2.2231   0.0657   4.4512  -0.4980  -0.1969   0.9826
#>   -9.6744 -10.1041  -4.7983   6.2825  -7.3330 -15.7613 -13.7212   5.0647
#>   -0.6882   0.0963  -5.8560   7.0549  -1.2726   2.0795   1.6458 -10.5091
#>    6.0436   0.3458  -8.0833 -16.3053  -4.4579  -6.6254  -2.9359  -4.0751
#>   -1.7251   5.0475  -4.1154  -3.6474   0.6908  10.1572  -2.7401  -5.7063
#>    1.4034   1.4782   0.9438   0.5476   8.5959  10.6777  -1.8240   5.6944
#>    3.1727  18.9440  -1.7049  -2.6877   4.5660  -3.4674   1.2797  -0.8448
#>    1.3719   8.5737  14.3520 -13.3018  -7.4383   6.1442   1.6040   2.2899
#>    1.2055  -9.9962  -8.6463   8.6708   7.5850  -9.7755   6.7613  -7.9331
#>    8.3799  21.1558   7.0549   2.6935   4.0890  -1.0101  -5.3987   3.3712
#>   -3.6468  18.7492  -0.1738  -5.1937   3.8317   9.0657   2.5692  10.6031
#>   -2.1354   9.6714  -3.9801 -11.2625   6.0492  -6.7784   4.5728   5.7826
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,48} ]
```
