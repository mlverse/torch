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
#>  1.2637 -0.2368
#>   0.2802  0.5236
#> 
#> (1,2,.,.) = 
#> -0.9607 -0.4491
#>   0.6514 -0.2698
#> 
#> (1,3,.,.) = 
#>  1.0411  1.2900
#>   0.9913 -1.8151
#> 
#> (1,4,.,.) = 
#> -1.2802 -0.5766
#>   0.4495 -1.1900
#> [ CPUFloatType{1,4,2,2} ]
#> torch_tensor
#> (1,1,.,.) = 
#>  1.2637 -0.2368
#>   0.2802  0.5236
#> 
#> (1,2,.,.) = 
#>  1.0411  1.2900
#>   0.9913 -1.8151
#> 
#> (1,3,.,.) = 
#> -0.9607 -0.4491
#>   0.6514 -0.2698
#> 
#> (1,4,.,.) = 
#> -1.2802 -0.5766
#>   0.4495 -1.1900
#> [ CPUFloatType{1,4,2,2} ]
```
