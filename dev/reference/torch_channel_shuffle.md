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
#>  1.5909 -0.1353
#>   0.7627 -2.1981
#> 
#> (1,2,.,.) = 
#>  0.4454 -1.3818
#>  -0.4092 -2.2921
#> 
#> (1,3,.,.) = 
#> -0.2964 -0.5454
#>   0.3141  2.3105
#> 
#> (1,4,.,.) = 
#> -0.2261 -0.9669
#>   2.4277  0.5852
#> [ CPUFloatType{1,4,2,2} ]
#> torch_tensor
#> (1,1,.,.) = 
#>  1.5909 -0.1353
#>   0.7627 -2.1981
#> 
#> (1,2,.,.) = 
#> -0.2964 -0.5454
#>   0.3141  2.3105
#> 
#> (1,3,.,.) = 
#>  0.4454 -1.3818
#>  -0.4092 -2.2921
#> 
#> (1,4,.,.) = 
#> -0.2261 -0.9669
#>   2.4277  0.5852
#> [ CPUFloatType{1,4,2,2} ]
```
