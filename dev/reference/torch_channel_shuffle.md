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
#> -0.7679 -1.5535
#>  -2.9857 -0.9581
#> 
#> (1,2,.,.) = 
#>  1.0514 -0.1752
#>   0.7754  0.3343
#> 
#> (1,3,.,.) = 
#> -1.6358 -1.0850
#>  -0.8066  0.7712
#> 
#> (1,4,.,.) = 
#>  0.3635 -0.9343
#>   1.0259  1.0034
#> [ CPUFloatType{1,4,2,2} ]
#> torch_tensor
#> (1,1,.,.) = 
#> -0.7679 -1.5535
#>  -2.9857 -0.9581
#> 
#> (1,2,.,.) = 
#> -1.6358 -1.0850
#>  -0.8066  0.7712
#> 
#> (1,3,.,.) = 
#>  1.0514 -0.1752
#>   0.7754  0.3343
#> 
#> (1,4,.,.) = 
#>  0.3635 -0.9343
#>   1.0259  1.0034
#> [ CPUFloatType{1,4,2,2} ]
```
