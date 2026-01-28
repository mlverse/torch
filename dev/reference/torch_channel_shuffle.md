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
#>   1.0585  0.5546
#>   0.3276 -0.0299
#> 
#> (1,2,.,.) = 
#>  -1.9318  1.7340
#>  -1.8660 -0.0455
#> 
#> (1,3,.,.) = 
#>  -1.3935  1.0309
#>  -1.3772 -0.1257
#> 
#> (1,4,.,.) = 
#>  -0.7952  0.0387
#>  -0.1631 -0.3129
#> [ CPUFloatType{1,4,2,2} ]
#> torch_tensor
#> (1,1,.,.) = 
#>   1.0585  0.5546
#>   0.3276 -0.0299
#> 
#> (1,2,.,.) = 
#>  -1.3935  1.0309
#>  -1.3772 -0.1257
#> 
#> (1,3,.,.) = 
#>  -1.9318  1.7340
#>  -1.8660 -0.0455
#> 
#> (1,4,.,.) = 
#>  -0.7952  0.0387
#>  -0.1631 -0.3129
#> [ CPUFloatType{1,4,2,2} ]
```
