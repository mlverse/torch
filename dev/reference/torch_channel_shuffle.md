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
#> -0.2201  1.0981
#>   0.1320 -1.3732
#> 
#> (1,2,.,.) = 
#> -0.2590  1.1114
#>   0.3782 -0.7573
#> 
#> (1,3,.,.) = 
#> -0.0554  1.8498
#>   0.0726 -1.7561
#> 
#> (1,4,.,.) = 
#>  0.1993 -0.1184
#>  -1.4703 -0.6075
#> [ CPUFloatType{1,4,2,2} ]
#> torch_tensor
#> (1,1,.,.) = 
#> -0.2201  1.0981
#>   0.1320 -1.3732
#> 
#> (1,2,.,.) = 
#> -0.0554  1.8498
#>   0.0726 -1.7561
#> 
#> (1,3,.,.) = 
#> -0.2590  1.1114
#>   0.3782 -0.7573
#> 
#> (1,4,.,.) = 
#>  0.1993 -0.1184
#>  -1.4703 -0.6075
#> [ CPUFloatType{1,4,2,2} ]
```
