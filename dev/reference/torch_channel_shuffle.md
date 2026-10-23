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
#> -0.1012 -1.1156
#>  -0.5841 -1.5765
#> 
#> (1,2,.,.) = 
#>  1.1702 -0.9770
#>   0.4734  0.6105
#> 
#> (1,3,.,.) = 
#> -0.7106 -0.3758
#>   0.1947  1.8130
#> 
#> (1,4,.,.) = 
#> -0.3078  1.6849
#>   0.6202  1.0341
#> [ CPUFloatType{1,4,2,2} ]
#> torch_tensor
#> (1,1,.,.) = 
#> -0.1012 -1.1156
#>  -0.5841 -1.5765
#> 
#> (1,2,.,.) = 
#> -0.7106 -0.3758
#>   0.1947  1.8130
#> 
#> (1,3,.,.) = 
#>  1.1702 -0.9770
#>   0.4734  0.6105
#> 
#> (1,4,.,.) = 
#> -0.3078  1.6849
#>   0.6202  1.0341
#> [ CPUFloatType{1,4,2,2} ]
```
