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
#> -1.2887  0.7674
#>  -1.0891 -0.5106
#> 
#> (1,2,.,.) = 
#>  0.3304  0.2258
#>   0.5868  0.3645
#> 
#> (1,3,.,.) = 
#> -0.3770  1.7214
#>   1.0536 -0.2430
#> 
#> (1,4,.,.) = 
#>  0.7565  0.1899
#>   1.2045  1.2089
#> [ CPUFloatType{1,4,2,2} ]
#> torch_tensor
#> (1,1,.,.) = 
#> -1.2887  0.7674
#>  -1.0891 -0.5106
#> 
#> (1,2,.,.) = 
#> -0.3770  1.7214
#>   1.0536 -0.2430
#> 
#> (1,3,.,.) = 
#>  0.3304  0.2258
#>   0.5868  0.3645
#> 
#> (1,4,.,.) = 
#>  0.7565  0.1899
#>   1.2045  1.2089
#> [ CPUFloatType{1,4,2,2} ]
```
