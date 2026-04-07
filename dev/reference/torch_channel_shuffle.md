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
#> -0.3936  0.2015
#>   0.2520 -0.6531
#> 
#> (1,2,.,.) = 
#> -1.8161  0.6450
#>  -0.8465  0.7176
#> 
#> (1,3,.,.) = 
#> -0.4018 -0.3590
#>   0.9216 -0.1654
#> 
#> (1,4,.,.) = 
#>  0.6465 -1.0172
#>   1.0482 -0.3565
#> [ CPUFloatType{1,4,2,2} ]
#> torch_tensor
#> (1,1,.,.) = 
#> -0.3936  0.2015
#>   0.2520 -0.6531
#> 
#> (1,2,.,.) = 
#> -0.4018 -0.3590
#>   0.9216 -0.1654
#> 
#> (1,3,.,.) = 
#> -1.8161  0.6450
#>  -0.8465  0.7176
#> 
#> (1,4,.,.) = 
#>  0.6465 -1.0172
#>   1.0482 -0.3565
#> [ CPUFloatType{1,4,2,2} ]
```
