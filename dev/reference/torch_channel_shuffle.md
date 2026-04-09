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
#> -1.1348  0.3734
#>  -0.2021 -1.1289
#> 
#> (1,2,.,.) = 
#> -0.1772 -0.2838
#>   1.1012  0.9304
#> 
#> (1,3,.,.) = 
#> -0.4395 -1.2588
#>  -1.0410  0.4048
#> 
#> (1,4,.,.) = 
#>  0.1187  0.9253
#>   0.0743  1.3015
#> [ CPUFloatType{1,4,2,2} ]
#> torch_tensor
#> (1,1,.,.) = 
#> -1.1348  0.3734
#>  -0.2021 -1.1289
#> 
#> (1,2,.,.) = 
#> -0.4395 -1.2588
#>  -1.0410  0.4048
#> 
#> (1,3,.,.) = 
#> -0.1772 -0.2838
#>   1.1012  0.9304
#> 
#> (1,4,.,.) = 
#>  0.1187  0.9253
#>   0.0743  1.3015
#> [ CPUFloatType{1,4,2,2} ]
```
