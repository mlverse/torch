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
#>   0.3253 -0.9913
#>  -0.9275  0.2362
#> 
#> (1,2,.,.) = 
#>   0.5050 -0.1119
#>  -0.3709 -0.5468
#> 
#> (1,3,.,.) = 
#>   0.9325 -0.0473
#>  -1.3015  0.4513
#> 
#> (1,4,.,.) = 
#>   1.0115  0.8366
#>  -0.1767  0.8246
#> [ CPUFloatType{1,4,2,2} ]
#> torch_tensor
#> (1,1,.,.) = 
#>   0.3253 -0.9913
#>  -0.9275  0.2362
#> 
#> (1,2,.,.) = 
#>   0.9325 -0.0473
#>  -1.3015  0.4513
#> 
#> (1,3,.,.) = 
#>   0.5050 -0.1119
#>  -0.3709 -0.5468
#> 
#> (1,4,.,.) = 
#>   1.0115  0.8366
#>  -0.1767  0.8246
#> [ CPUFloatType{1,4,2,2} ]
```
