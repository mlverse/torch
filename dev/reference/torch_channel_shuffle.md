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
#>  0.5561  1.1723
#>  -1.8056  0.6881
#> 
#> (1,2,.,.) = 
#>  1.3689 -0.3948
#>   0.6782 -0.9710
#> 
#> (1,3,.,.) = 
#> -0.1198 -0.4379
#>   0.6270  0.1410
#> 
#> (1,4,.,.) = 
#> -0.3357  0.3290
#>  -0.5720  0.8849
#> [ CPUFloatType{1,4,2,2} ]
#> torch_tensor
#> (1,1,.,.) = 
#>  0.5561  1.1723
#>  -1.8056  0.6881
#> 
#> (1,2,.,.) = 
#> -0.1198 -0.4379
#>   0.6270  0.1410
#> 
#> (1,3,.,.) = 
#>  1.3689 -0.3948
#>   0.6782 -0.9710
#> 
#> (1,4,.,.) = 
#> -0.3357  0.3290
#>  -0.5720  0.8849
#> [ CPUFloatType{1,4,2,2} ]
```
