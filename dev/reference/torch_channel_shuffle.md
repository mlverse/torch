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
#>   0.6449  2.1436
#>  -0.2683 -0.1350
#> 
#> (1,2,.,.) = 
#>   0.0453  0.3794
#>  -0.9156  0.1489
#> 
#> (1,3,.,.) = 
#>   1.3362  0.6405
#>   0.2725  0.6336
#> 
#> (1,4,.,.) = 
#>  -0.1447 -0.1773
#>  -1.7023 -0.2493
#> [ CPUFloatType{1,4,2,2} ]
#> torch_tensor
#> (1,1,.,.) = 
#>   0.6449  2.1436
#>  -0.2683 -0.1350
#> 
#> (1,2,.,.) = 
#>   1.3362  0.6405
#>   0.2725  0.6336
#> 
#> (1,3,.,.) = 
#>   0.0453  0.3794
#>  -0.9156  0.1489
#> 
#> (1,4,.,.) = 
#>  -0.1447 -0.1773
#>  -1.7023 -0.2493
#> [ CPUFloatType{1,4,2,2} ]
```
