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
#> -0.1375  0.3265
#>   0.1042  0.0516
#> 
#> (1,2,.,.) = 
#>  0.2089  0.5505
#>   0.0847  1.3241
#> 
#> (1,3,.,.) = 
#>  1.0610 -2.4817
#>   1.0179 -0.2126
#> 
#> (1,4,.,.) = 
#> -0.4293 -0.1001
#>  -1.2460  0.3920
#> [ CPUFloatType{1,4,2,2} ]
#> torch_tensor
#> (1,1,.,.) = 
#> -0.1375  0.3265
#>   0.1042  0.0516
#> 
#> (1,2,.,.) = 
#>  1.0610 -2.4817
#>   1.0179 -0.2126
#> 
#> (1,3,.,.) = 
#>  0.2089  0.5505
#>   0.0847  1.3241
#> 
#> (1,4,.,.) = 
#> -0.4293 -0.1001
#>  -1.2460  0.3920
#> [ CPUFloatType{1,4,2,2} ]
```
