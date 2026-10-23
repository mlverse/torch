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
#> -0.3441  0.5121
#>  -0.5487 -1.5494
#> 
#> (1,2,.,.) = 
#> -0.6399 -1.5240
#>   0.2923 -1.2562
#> 
#> (1,3,.,.) = 
#> -0.9021 -0.5934
#>  -0.4385  0.6543
#> 
#> (1,4,.,.) = 
#> -0.4983  1.6527
#>  -0.2223  0.7020
#> [ CPUFloatType{1,4,2,2} ]
#> torch_tensor
#> (1,1,.,.) = 
#> -0.3441  0.5121
#>  -0.5487 -1.5494
#> 
#> (1,2,.,.) = 
#> -0.9021 -0.5934
#>  -0.4385  0.6543
#> 
#> (1,3,.,.) = 
#> -0.6399 -1.5240
#>   0.2923 -1.2562
#> 
#> (1,4,.,.) = 
#> -0.4983  1.6527
#>  -0.2223  0.7020
#> [ CPUFloatType{1,4,2,2} ]
```
