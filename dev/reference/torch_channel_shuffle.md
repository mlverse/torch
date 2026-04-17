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
#> -1.5017  1.5260
#>   1.8294  1.6172
#> 
#> (1,2,.,.) = 
#> -0.5061  0.0853
#>  -1.0266 -0.0320
#> 
#> (1,3,.,.) = 
#> -1.0948  0.2334
#>   0.6765  0.7252
#> 
#> (1,4,.,.) = 
#>  0.1811  1.5031
#>   1.2694  0.1804
#> [ CPUFloatType{1,4,2,2} ]
#> torch_tensor
#> (1,1,.,.) = 
#> -1.5017  1.5260
#>   1.8294  1.6172
#> 
#> (1,2,.,.) = 
#> -1.0948  0.2334
#>   0.6765  0.7252
#> 
#> (1,3,.,.) = 
#> -0.5061  0.0853
#>  -1.0266 -0.0320
#> 
#> (1,4,.,.) = 
#>  0.1811  1.5031
#>   1.2694  0.1804
#> [ CPUFloatType{1,4,2,2} ]
```
