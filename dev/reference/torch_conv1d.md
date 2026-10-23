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
#> Columns 1 to 8  -4.0219  -6.4101  -2.7715   4.4499   4.4712  -4.6122 -11.3554  -2.1047
#>    4.8424   0.4107   0.8091  -7.5339  -3.6651  -1.7610  -2.0651   1.0495
#>   -4.5602  -4.4658  10.9486  -5.5690  -4.8477  11.3153   4.4407  -0.0457
#>   -0.6995   2.6401   1.5141  -6.6614  -7.6155  -8.8346   7.6124   6.4053
#>   -0.6126   0.7250   3.0152  -3.8954   2.9375   0.8148  11.0160  -0.0084
#>    0.4822  -6.3742   5.0712  -4.6123  -5.6153   0.8715   0.5230  -6.4041
#>    2.8136  -8.8246   2.9568  -6.6909  -8.9940  -0.3319  -0.7693  10.9599
#>    2.3120  11.2949  -3.6298   7.0008   8.2122   5.7774   0.3619   1.3600
#>   -2.6572  -0.2743 -11.6434   1.0535  10.0397  14.3547   4.4974   0.8820
#>   -6.0237   2.6095 -11.8931  -8.6899   5.4662   8.2232   4.0539  13.5279
#>   -1.3362   8.1279   7.1912   5.1849   0.9120   0.8365   2.2528  -5.3174
#>   -5.7027   2.4001  10.7831   5.5315  -8.8134   4.3555  -4.7336   1.0916
#>   -4.1054  -8.9371   0.7980   6.0149  -0.3609   4.5846  -8.2053   0.8153
#>    5.1028  11.0044  -9.9435 -13.5635  -6.3763  -0.4327  -4.8166  10.8311
#>  -10.2100   6.4441  -0.8796  -4.5629 -10.4438  -6.5035   5.5771  -5.2689
#>   -2.0307  -2.0064   6.4812  -0.1256  -6.7065  -8.8361  -4.5699   3.9091
#>  -14.9485  -3.3449  -3.3424  -4.2111  -8.8606   7.6645   1.9475  -1.3096
#>    0.3952  -3.1676  -9.2160   3.6319   6.0207  -6.0132  -4.3045  -8.4084
#>    3.9654  -7.2516  -0.3875   8.3230  -0.3950   6.2114  -4.7772  -0.7252
#>   -0.9126  -2.6217  -6.3317   1.6287   1.9093  -4.7110  17.4521   4.9732
#>  -13.8451   3.4603 -16.8490 -12.2213   2.6470  13.3799   9.0957   5.0275
#>   -6.5851  -7.0961  -2.1073 -13.0897  -0.8512   1.1524   5.3819  -1.8925
#>   -0.4053  -2.2407  -2.6127  -3.7069   7.3327  -0.5829 -15.2999  -1.0229
#>  -10.2432  -5.1767   2.0145 -14.7842  -0.4767  13.4658   9.2932  -1.7176
#>    0.8184   3.4310  12.4142   0.5330  -5.9949  -1.9785  11.4704   4.8497
#>    4.8752   2.8219  -2.0178  -2.4741  -8.2294 -11.9420  -1.4417  -4.6947
#>    5.8859   0.0076   7.5567   0.1542  -8.3382   2.5619   0.9395  -4.8024
#>    3.1823   1.5151  -1.1857   7.0197  -4.8558   0.3492  -1.4731  -3.1757
#>   -6.0715   4.6083  -4.0791   6.8838   0.5378  -0.3403  15.1969  -2.9437
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{20,33,48} ]
```
