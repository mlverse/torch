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
#>   0.7207  1.5883
#>  -0.1177 -0.7065
#> 
#> (1,2,.,.) = 
#>  -0.6032 -1.0536
#>   0.3501  0.4325
#> 
#> (1,3,.,.) = 
#>  -0.1187 -0.2967
#>  -0.6282 -0.1697
#> 
#> (1,4,.,.) = 
#>   0.8390 -1.5291
#>   0.3851  1.2488
#> [ CPUFloatType{1,4,2,2} ]
#> torch_tensor
#> (1,1,.,.) = 
#>   0.7207  1.5883
#>  -0.1177 -0.7065
#> 
#> (1,2,.,.) = 
#>  -0.1187 -0.2967
#>  -0.6282 -0.1697
#> 
#> (1,3,.,.) = 
#>  -0.6032 -1.0536
#>   0.3501  0.4325
#> 
#> (1,4,.,.) = 
#>   0.8390 -1.5291
#>   0.3851  1.2488
#> [ CPUFloatType{1,4,2,2} ]
```
