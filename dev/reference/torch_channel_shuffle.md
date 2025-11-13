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
#>   0.7183 -0.8660
#>  -1.1688  0.7744
#> 
#> (1,2,.,.) = 
#>  -1.5479  0.8965
#>  -0.8973  0.4180
#> 
#> (1,3,.,.) = 
#>   0.4655 -1.0279
#>   0.9086 -0.5825
#> 
#> (1,4,.,.) = 
#>   1.7017 -0.0396
#>   1.0564  0.3593
#> [ CPUFloatType{1,4,2,2} ]
#> torch_tensor
#> (1,1,.,.) = 
#>   0.7183 -0.8660
#>  -1.1688  0.7744
#> 
#> (1,2,.,.) = 
#>   0.4655 -1.0279
#>   0.9086 -0.5825
#> 
#> (1,3,.,.) = 
#>  -1.5479  0.8965
#>  -0.8973  0.4180
#> 
#> (1,4,.,.) = 
#>   1.7017 -0.0396
#>   1.0564  0.3593
#> [ CPUFloatType{1,4,2,2} ]
```
