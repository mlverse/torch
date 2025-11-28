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
#>   0.3506  0.4429
#>   1.3645 -0.0360
#> 
#> (1,2,.,.) = 
#>  -0.9173 -1.4383
#>  -1.0997 -0.9992
#> 
#> (1,3,.,.) = 
#>   1.3899  1.8867
#>  -0.4853 -1.5983
#> 
#> (1,4,.,.) = 
#>  -1.1643  0.5765
#>  -0.3170 -1.3633
#> [ CPUFloatType{1,4,2,2} ]
#> torch_tensor
#> (1,1,.,.) = 
#>   0.3506  0.4429
#>   1.3645 -0.0360
#> 
#> (1,2,.,.) = 
#>   1.3899  1.8867
#>  -0.4853 -1.5983
#> 
#> (1,3,.,.) = 
#>  -0.9173 -1.4383
#>  -1.0997 -0.9992
#> 
#> (1,4,.,.) = 
#>  -1.1643  0.5765
#>  -0.3170 -1.3633
#> [ CPUFloatType{1,4,2,2} ]
```
