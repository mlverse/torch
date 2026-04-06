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
#> -0.0824 -0.8330
#>   0.1384 -0.4018
#> 
#> (1,2,.,.) = 
#>  0.9873 -0.9288
#>  -1.8268  0.1412
#> 
#> (1,3,.,.) = 
#> -0.3970 -0.4782
#>  -1.9593  0.5219
#> 
#> (1,4,.,.) = 
#> -0.6424 -0.1264
#>   0.7075 -0.8503
#> [ CPUFloatType{1,4,2,2} ]
#> torch_tensor
#> (1,1,.,.) = 
#> -0.0824 -0.8330
#>   0.1384 -0.4018
#> 
#> (1,2,.,.) = 
#> -0.3970 -0.4782
#>  -1.9593  0.5219
#> 
#> (1,3,.,.) = 
#>  0.9873 -0.9288
#>  -1.8268  0.1412
#> 
#> (1,4,.,.) = 
#> -0.6424 -0.1264
#>   0.7075 -0.8503
#> [ CPUFloatType{1,4,2,2} ]
```
