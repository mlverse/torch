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
#>  -0.5448 -0.3395
#>  -0.9064  0.1848
#> 
#> (1,2,.,.) = 
#>   0.3671 -0.2383
#>  -0.6793 -1.0023
#> 
#> (1,3,.,.) = 
#>   0.6853  0.1393
#>   0.0744 -0.2894
#> 
#> (1,4,.,.) = 
#>  -1.0058  0.5430
#>  -0.2802  0.6543
#> [ CPUFloatType{1,4,2,2} ]
#> torch_tensor
#> (1,1,.,.) = 
#>  -0.5448 -0.3395
#>  -0.9064  0.1848
#> 
#> (1,2,.,.) = 
#>   0.6853  0.1393
#>   0.0744 -0.2894
#> 
#> (1,3,.,.) = 
#>   0.3671 -0.2383
#>  -0.6793 -1.0023
#> 
#> (1,4,.,.) = 
#>  -1.0058  0.5430
#>  -0.2802  0.6543
#> [ CPUFloatType{1,4,2,2} ]
```
