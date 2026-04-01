# Channel_shuffle

Channel_shuffle

## Usage

``` r
torch_channel_shuffle(self, groups)
```

## Arguments

- self:

  (Tensor) the input tensor

- groups:

  (int) number of groups to divide channels in and rearrange.

## Divide the channels in a tensor of shape

math:`(*, C , H, W)` :

Divide the channels in a tensor of shape \\(\*, C , H, W)\\ into g
groups and rearrange them as \\(\*, C \frac g, g, H, W)\\, while keeping
the original tensor shape.

## Examples

``` r
if (torch_is_installed()) {

input <- torch_randn(c(1, 4, 2, 2))
print(input)
output <- torch_channel_shuffle(input, 2)
print(output)
}
#> torch_tensor
#> (1,1,.,.) = 
#>   1.6091 -0.2505
#>   1.8359  1.6453
#> 
#> (1,2,.,.) = 
#>  -1.1136  0.5490
#>   0.1794  0.6477
#> 
#> (1,3,.,.) = 
#>  -0.4215 -0.6142
#>   1.1628  0.1459
#> 
#> (1,4,.,.) = 
#>   1.1320 -0.2709
#>   0.8658 -2.3362
#> [ CPUFloatType{1,4,2,2} ]
#> torch_tensor
#> (1,1,.,.) = 
#>   1.6091 -0.2505
#>   1.8359  1.6453
#> 
#> (1,2,.,.) = 
#>  -0.4215 -0.6142
#>   1.1628  0.1459
#> 
#> (1,3,.,.) = 
#>  -1.1136  0.5490
#>   0.1794  0.6477
#> 
#> (1,4,.,.) = 
#>   1.1320 -0.2709
#>   0.8658 -2.3362
#> [ CPUFloatType{1,4,2,2} ]
```
