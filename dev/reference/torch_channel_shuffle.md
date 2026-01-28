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
#>   0.1783  0.9890
#>  -0.7586  0.9680
#> 
#> (1,2,.,.) = 
#>  -1.0342  0.3631
#>  -1.0191 -0.2648
#> 
#> (1,3,.,.) = 
#>  -0.5852  0.2941
#>   0.8834 -0.0358
#> 
#> (1,4,.,.) = 
#>  -0.1389 -0.6541
#>   0.9270 -0.3876
#> [ CPUFloatType{1,4,2,2} ]
#> torch_tensor
#> (1,1,.,.) = 
#>   0.1783  0.9890
#>  -0.7586  0.9680
#> 
#> (1,2,.,.) = 
#>  -0.5852  0.2941
#>   0.8834 -0.0358
#> 
#> (1,3,.,.) = 
#>  -1.0342  0.3631
#>  -1.0191 -0.2648
#> 
#> (1,4,.,.) = 
#>  -0.1389 -0.6541
#>   0.9270 -0.3876
#> [ CPUFloatType{1,4,2,2} ]
```
