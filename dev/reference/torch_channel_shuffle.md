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
#> -0.6110  0.2761
#>   0.9769  0.4407
#> 
#> (1,2,.,.) = 
#> -0.3066 -0.2266
#>  -0.3653 -1.4741
#> 
#> (1,3,.,.) = 
#>  0.7277 -0.9508
#>   0.3449  0.7738
#> 
#> (1,4,.,.) = 
#> -1.5371 -0.1314
#>  -1.0223  0.3131
#> [ CPUFloatType{1,4,2,2} ]
#> torch_tensor
#> (1,1,.,.) = 
#> -0.6110  0.2761
#>   0.9769  0.4407
#> 
#> (1,2,.,.) = 
#>  0.7277 -0.9508
#>   0.3449  0.7738
#> 
#> (1,3,.,.) = 
#> -0.3066 -0.2266
#>  -0.3653 -1.4741
#> 
#> (1,4,.,.) = 
#> -1.5371 -0.1314
#>  -1.0223  0.3131
#> [ CPUFloatType{1,4,2,2} ]
```
