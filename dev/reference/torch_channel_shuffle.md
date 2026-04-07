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
#> -0.3005 -0.1736
#>  -1.3881 -1.1283
#> 
#> (1,2,.,.) = 
#>  0.5606 -0.9860
#>   0.3246  2.0506
#> 
#> (1,3,.,.) = 
#> -1.1684  1.2381
#>   0.0907  0.3044
#> 
#> (1,4,.,.) = 
#>  0.5678 -1.2230
#>   0.8643  0.8997
#> [ CPUFloatType{1,4,2,2} ]
#> torch_tensor
#> (1,1,.,.) = 
#> -0.3005 -0.1736
#>  -1.3881 -1.1283
#> 
#> (1,2,.,.) = 
#> -1.1684  1.2381
#>   0.0907  0.3044
#> 
#> (1,3,.,.) = 
#>  0.5606 -0.9860
#>   0.3246  2.0506
#> 
#> (1,4,.,.) = 
#>  0.5678 -1.2230
#>   0.8643  0.8997
#> [ CPUFloatType{1,4,2,2} ]
```
