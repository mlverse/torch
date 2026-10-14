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
#> -1.9950 -0.2325
#>  -0.5121 -0.4495
#> 
#> (1,2,.,.) = 
#>  0.7560  0.2056
#>  -0.4985 -0.0206
#> 
#> (1,3,.,.) = 
#> -0.4051  0.5907
#>  -0.7537 -0.2977
#> 
#> (1,4,.,.) = 
#> -2.0159  0.0193
#>   0.0080  0.6015
#> [ CPUFloatType{1,4,2,2} ]
#> torch_tensor
#> (1,1,.,.) = 
#> -1.9950 -0.2325
#>  -0.5121 -0.4495
#> 
#> (1,2,.,.) = 
#> -0.4051  0.5907
#>  -0.7537 -0.2977
#> 
#> (1,3,.,.) = 
#>  0.7560  0.2056
#>  -0.4985 -0.0206
#> 
#> (1,4,.,.) = 
#> -2.0159  0.0193
#>   0.0080  0.6015
#> [ CPUFloatType{1,4,2,2} ]
```
