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
#>  -0.8875  0.4346
#>  -0.8845  0.4485
#> 
#> (1,2,.,.) = 
#>  -0.0443 -0.8792
#>   0.7010 -0.4430
#> 
#> (1,3,.,.) = 
#>   0.7555  0.1266
#>  -0.0962 -0.1504
#> 
#> (1,4,.,.) = 
#>  -0.4055  0.3781
#>  -1.3313 -1.1470
#> [ CPUFloatType{1,4,2,2} ]
#> torch_tensor
#> (1,1,.,.) = 
#>  -0.8875  0.4346
#>  -0.8845  0.4485
#> 
#> (1,2,.,.) = 
#>   0.7555  0.1266
#>  -0.0962 -0.1504
#> 
#> (1,3,.,.) = 
#>  -0.0443 -0.8792
#>   0.7010 -0.4430
#> 
#> (1,4,.,.) = 
#>  -0.4055  0.3781
#>  -1.3313 -1.1470
#> [ CPUFloatType{1,4,2,2} ]
```
