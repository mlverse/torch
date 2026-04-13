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
#> -0.4226  0.4756
#>   0.6937  0.5312
#> 
#> (1,2,.,.) = 
#>  0.0970 -0.3326
#>   0.1229  0.1174
#> 
#> (1,3,.,.) = 
#> -0.1994 -0.8148
#>   0.6304 -0.8734
#> 
#> (1,4,.,.) = 
#>  0.8496  1.4453
#>  -0.5919 -0.1708
#> [ CPUFloatType{1,4,2,2} ]
#> torch_tensor
#> (1,1,.,.) = 
#> -0.4226  0.4756
#>   0.6937  0.5312
#> 
#> (1,2,.,.) = 
#> -0.1994 -0.8148
#>   0.6304 -0.8734
#> 
#> (1,3,.,.) = 
#>  0.0970 -0.3326
#>   0.1229  0.1174
#> 
#> (1,4,.,.) = 
#>  0.8496  1.4453
#>  -0.5919 -0.1708
#> [ CPUFloatType{1,4,2,2} ]
```
