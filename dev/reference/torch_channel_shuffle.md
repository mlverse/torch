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
#>  0.4971  0.6597
#>   2.1970 -1.0547
#> 
#> (1,2,.,.) = 
#>  2.0178  2.6442
#>  -0.1969 -1.1168
#> 
#> (1,3,.,.) = 
#> -1.4352  2.8991
#>   0.2429 -0.0439
#> 
#> (1,4,.,.) = 
#>  0.2324  1.1144
#>   0.7082  1.1646
#> [ CPUFloatType{1,4,2,2} ]
#> torch_tensor
#> (1,1,.,.) = 
#>  0.4971  0.6597
#>   2.1970 -1.0547
#> 
#> (1,2,.,.) = 
#> -1.4352  2.8991
#>   0.2429 -0.0439
#> 
#> (1,3,.,.) = 
#>  2.0178  2.6442
#>  -0.1969 -1.1168
#> 
#> (1,4,.,.) = 
#>  0.2324  1.1144
#>   0.7082  1.1646
#> [ CPUFloatType{1,4,2,2} ]
```
