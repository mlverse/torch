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
#> -0.2578  0.2522
#>   1.1506  0.1624
#> 
#> (1,2,.,.) = 
#>  0.0530  0.6530
#>   0.3552 -0.3587
#> 
#> (1,3,.,.) = 
#> -1.8790  0.5684
#>  -0.0373  0.3957
#> 
#> (1,4,.,.) = 
#> -0.4855 -0.1820
#>   0.9762  0.8227
#> [ CPUFloatType{1,4,2,2} ]
#> torch_tensor
#> (1,1,.,.) = 
#> -0.2578  0.2522
#>   1.1506  0.1624
#> 
#> (1,2,.,.) = 
#> -1.8790  0.5684
#>  -0.0373  0.3957
#> 
#> (1,3,.,.) = 
#>  0.0530  0.6530
#>   0.3552 -0.3587
#> 
#> (1,4,.,.) = 
#> -0.4855 -0.1820
#>   0.9762  0.8227
#> [ CPUFloatType{1,4,2,2} ]
```
