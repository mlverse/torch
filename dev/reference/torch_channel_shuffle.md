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
#>   0.4092  0.9040
#>   0.0750 -0.2560
#> 
#> (1,2,.,.) = 
#>  0.01 *
#>   2.6442 -0.3095
#>   -42.4806 -14.3095
#> 
#> (1,3,.,.) = 
#>   0.6985  0.1305
#>  -1.3049  0.2057
#> 
#> (1,4,.,.) = 
#>  -0.5565  1.1565
#>   0.1293  1.1955
#> [ CPUFloatType{1,4,2,2} ]
#> torch_tensor
#> (1,1,.,.) = 
#>   0.4092  0.9040
#>   0.0750 -0.2560
#> 
#> (1,2,.,.) = 
#>   0.6985  0.1305
#>  -1.3049  0.2057
#> 
#> (1,3,.,.) = 
#>  0.01 *
#>   2.6442 -0.3095
#>   -42.4806 -14.3095
#> 
#> (1,4,.,.) = 
#>  -0.5565  1.1565
#>   0.1293  1.1955
#> [ CPUFloatType{1,4,2,2} ]
```
