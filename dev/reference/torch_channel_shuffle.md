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
#>  1.6903 -0.5616
#>  -0.1026  0.7208
#> 
#> (1,2,.,.) = 
#> -0.9234 -1.3111
#>  -0.0645 -0.2905
#> 
#> (1,3,.,.) = 
#>  0.1582  0.9218
#>  -0.2400 -1.0573
#> 
#> (1,4,.,.) = 
#> -0.0968 -1.7894
#>  -1.4152 -0.8761
#> [ CPUFloatType{1,4,2,2} ]
#> torch_tensor
#> (1,1,.,.) = 
#>  1.6903 -0.5616
#>  -0.1026  0.7208
#> 
#> (1,2,.,.) = 
#>  0.1582  0.9218
#>  -0.2400 -1.0573
#> 
#> (1,3,.,.) = 
#> -0.9234 -1.3111
#>  -0.0645 -0.2905
#> 
#> (1,4,.,.) = 
#> -0.0968 -1.7894
#>  -1.4152 -0.8761
#> [ CPUFloatType{1,4,2,2} ]
```
