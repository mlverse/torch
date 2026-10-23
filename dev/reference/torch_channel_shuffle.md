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
#>  0.1655 -0.9973
#>   0.7907  1.0943
#> 
#> (1,2,.,.) = 
#> -2.0593  1.5272
#>   0.4987 -0.1401
#> 
#> (1,3,.,.) = 
#>  0.4093  1.8004
#>   0.3792  1.3795
#> 
#> (1,4,.,.) = 
#>  0.7437  0.5868
#>  -1.1242 -0.3053
#> [ CPUFloatType{1,4,2,2} ]
#> torch_tensor
#> (1,1,.,.) = 
#>  0.1655 -0.9973
#>   0.7907  1.0943
#> 
#> (1,2,.,.) = 
#>  0.4093  1.8004
#>   0.3792  1.3795
#> 
#> (1,3,.,.) = 
#> -2.0593  1.5272
#>   0.4987 -0.1401
#> 
#> (1,4,.,.) = 
#>  0.7437  0.5868
#>  -1.1242 -0.3053
#> [ CPUFloatType{1,4,2,2} ]
```
