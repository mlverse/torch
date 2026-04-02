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
#>  Columns 1 to 8  11.1703   3.3670  -9.3786  -0.9654 -16.1547  -4.3128  -6.9850  -7.3555
#>    9.7815   5.1548   1.2192   6.3017   3.3265  10.3147  -0.1028   9.5770
#>   -2.5527   8.6107  12.4515  -2.1125   1.1627  -1.3164   4.5987   2.6987
#>   -5.9281   4.3542  -5.0552   5.0320 -12.0758   0.3133   2.4718  -1.2674
#>   -2.2058   8.1592   5.9470   6.7785   1.5439  -2.0083   1.1508  -5.2247
#>   12.9261   0.1598   1.0421  -8.6969   8.1363  -3.8587   5.8926  -0.0804
#>   22.0345   1.7200  -8.9608  12.3377 -20.8879   3.8800   4.7905  -0.3516
#>   -6.9947  -4.3773   5.0990  -4.3378   4.0865   3.5650   0.0887  -6.0947
#>    7.3729  -1.5029  -5.6533  -1.2638  -2.1924   0.8423  -3.0886  -3.1484
#>   -1.4229   1.2895  -7.7254   1.7227   2.2731  -4.6879  -2.7615   2.8952
#>   -5.7816  -9.2513  -0.2974  -8.8052   1.7347  -0.5269   0.3580 -10.5831
#>  -11.9957   3.0062   2.5606  -6.9433   8.9303   0.1080   5.5132   8.0333
#>  -10.2536  -4.6756  -7.3827   4.4003 -13.3057  -6.4228   2.9494  -8.9643
#>   -3.6955  -4.6171   4.7323 -10.2982   8.2436   1.5551   4.1173  -1.9457
#>  -14.7316   4.8832  -5.3329  -6.4149  12.5840  -3.6965  -6.9699  -3.6506
#>    4.8866  -2.7752   6.7710  10.6224  -1.2225  -3.5258  -2.6387   7.8583
#>   11.2125   1.4976  -3.9520   6.8514  11.2992   6.8914   0.8776   1.6768
#>    5.4860   3.4895  10.0873   1.7012 -13.1164   7.6977   0.9162   4.0777
#>   -7.1463  -4.9524   5.0235  -3.0411  -3.7134  -0.9816  -2.0670  -0.0047
#>   -7.6768   5.4800  12.9937  -6.0901  -1.7021  -3.3791   2.5170   0.9367
#>    2.4923   1.5668  -5.0036   3.3755  -1.7378  -4.7709  -0.9895  -2.3768
#>   -0.6391  -7.5086 -10.6867  10.1574  -5.8354   2.8010   0.4318   0.9592
#>   -3.8216  -2.8352   4.7089   0.0513 -11.4438   1.8674   2.2968  -4.7827
#>   -3.8971  -0.0514  -1.7790  -5.4431   1.5829  -6.9029  -0.2328  -8.6388
#>   -5.0678  -0.2822   4.0630  -5.4921  -6.9416   3.3978 -14.1884  -7.3870
#>    3.2039   2.3166 -10.2555   0.7879  -4.4552   7.6918   4.7459  -6.4730
#>   -1.1523  -0.0632  -8.3391  -4.5764   4.7779   4.5734  -4.5417  -6.6263
#>   12.9573   0.9868   5.7394  -0.9519   2.0689  10.5429  -4.9774   3.8884
#>   -6.2135   0.9569  -0.7070   0.8969 -11.0958   9.3686  -4.3995   2.9552
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,48} ]
```
