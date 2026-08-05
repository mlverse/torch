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
#>  0.8008 -1.2948
#>  -1.0557 -0.1235
#> 
#> (1,2,.,.) = 
#> -0.1828  0.9777
#>   0.0676  0.7988
#> 
#> (1,3,.,.) = 
#> -1.0415  0.4616
#>   1.0204 -0.6203
#> 
#> (1,4,.,.) = 
#>  0.9374  1.1305
#>   0.4197  1.3128
#> [ CPUFloatType{1,4,2,2} ]
#> torch_tensor
#> (1,1,.,.) = 
#>  0.8008 -1.2948
#>  -1.0557 -0.1235
#> 
#> (1,2,.,.) = 
#> -1.0415  0.4616
#>   1.0204 -0.6203
#> 
#> (1,3,.,.) = 
#> -0.1828  0.9777
#>   0.0676  0.7988
#> 
#> (1,4,.,.) = 
#>  0.9374  1.1305
#>   0.4197  1.3128
#> [ CPUFloatType{1,4,2,2} ]
```
