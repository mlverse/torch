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
#>  -1.3158 -1.3711
#>   0.1284 -0.2597
#> 
#> (1,2,.,.) = 
#>  -0.6471 -3.0265
#>   1.9080 -1.0513
#> 
#> (1,3,.,.) = 
#>   0.5751  0.9978
#>   0.4777 -1.3045
#> 
#> (1,4,.,.) = 
#>   0.8225 -0.5672
#>  -0.6554 -0.9732
#> [ CPUFloatType{1,4,2,2} ]
#> torch_tensor
#> (1,1,.,.) = 
#>  -1.3158 -1.3711
#>   0.1284 -0.2597
#> 
#> (1,2,.,.) = 
#>   0.5751  0.9978
#>   0.4777 -1.3045
#> 
#> (1,3,.,.) = 
#>  -0.6471 -3.0265
#>   1.9080 -1.0513
#> 
#> (1,4,.,.) = 
#>   0.8225 -0.5672
#>  -0.6554 -0.9732
#> [ CPUFloatType{1,4,2,2} ]
```
