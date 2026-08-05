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
#>  0.6719 -0.8688
#>   2.7937  0.4016
#> 
#> (1,2,.,.) = 
#>  0.6160 -0.9182
#>   0.4475 -1.0098
#> 
#> (1,3,.,.) = 
#>  0.2701 -1.5981
#>  -0.6555 -0.3025
#> 
#> (1,4,.,.) = 
#>  1.2833 -0.0995
#>   1.8610 -0.3560
#> [ CPUFloatType{1,4,2,2} ]
#> torch_tensor
#> (1,1,.,.) = 
#>  0.6719 -0.8688
#>   2.7937  0.4016
#> 
#> (1,2,.,.) = 
#>  0.2701 -1.5981
#>  -0.6555 -0.3025
#> 
#> (1,3,.,.) = 
#>  0.6160 -0.9182
#>   0.4475 -1.0098
#> 
#> (1,4,.,.) = 
#>  1.2833 -0.0995
#>   1.8610 -0.3560
#> [ CPUFloatType{1,4,2,2} ]
```
