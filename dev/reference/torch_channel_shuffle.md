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
#> -0.8453  1.6698
#>   0.5141  1.1415
#> 
#> (1,2,.,.) = 
#> -0.9493  1.6738
#>   0.1612 -0.2751
#> 
#> (1,3,.,.) = 
#> -0.1778 -1.4453
#>   0.6175 -1.0996
#> 
#> (1,4,.,.) = 
#> -1.0839 -0.0589
#>  -0.2048  0.6738
#> [ CPUFloatType{1,4,2,2} ]
#> torch_tensor
#> (1,1,.,.) = 
#> -0.8453  1.6698
#>   0.5141  1.1415
#> 
#> (1,2,.,.) = 
#> -0.1778 -1.4453
#>   0.6175 -1.0996
#> 
#> (1,3,.,.) = 
#> -0.9493  1.6738
#>   0.1612 -0.2751
#> 
#> (1,4,.,.) = 
#> -1.0839 -0.0589
#>  -0.2048  0.6738
#> [ CPUFloatType{1,4,2,2} ]
```
