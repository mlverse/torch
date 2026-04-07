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
#>  0.0910  1.5779
#>   1.2043  0.5485
#> 
#> (1,2,.,.) = 
#>  0.4735  0.0403
#>   0.8125  0.2296
#> 
#> (1,3,.,.) = 
#>  1.5817 -0.0238
#>   0.7105 -0.0553
#> 
#> (1,4,.,.) = 
#>  1.7415 -1.4138
#>   0.5197 -1.4922
#> [ CPUFloatType{1,4,2,2} ]
#> torch_tensor
#> (1,1,.,.) = 
#>  0.0910  1.5779
#>   1.2043  0.5485
#> 
#> (1,2,.,.) = 
#>  1.5817 -0.0238
#>   0.7105 -0.0553
#> 
#> (1,3,.,.) = 
#>  0.4735  0.0403
#>   0.8125  0.2296
#> 
#> (1,4,.,.) = 
#>  1.7415 -1.4138
#>   0.5197 -1.4922
#> [ CPUFloatType{1,4,2,2} ]
```
