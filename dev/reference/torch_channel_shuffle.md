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
#>  0.6217  0.4875
#>   0.1410 -0.6313
#> 
#> (1,2,.,.) = 
#> -0.6384  0.9780
#>  -1.0800  2.8643
#> 
#> (1,3,.,.) = 
#>  0.1558  2.0865
#>   1.3867 -0.8442
#> 
#> (1,4,.,.) = 
#>  0.8061 -0.6846
#>   1.3263  0.6307
#> [ CPUFloatType{1,4,2,2} ]
#> torch_tensor
#> (1,1,.,.) = 
#>  0.6217  0.4875
#>   0.1410 -0.6313
#> 
#> (1,2,.,.) = 
#>  0.1558  2.0865
#>   1.3867 -0.8442
#> 
#> (1,3,.,.) = 
#> -0.6384  0.9780
#>  -1.0800  2.8643
#> 
#> (1,4,.,.) = 
#>  0.8061 -0.6846
#>   1.3263  0.6307
#> [ CPUFloatType{1,4,2,2} ]
```
