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
#>  1.4042  0.5580
#>  -0.6112 -0.0664
#> 
#> (1,2,.,.) = 
#>  0.6423 -0.1072
#>  -0.0736  0.9928
#> 
#> (1,3,.,.) = 
#> -0.5515  0.0716
#>  -1.2813 -1.2546
#> 
#> (1,4,.,.) = 
#>  0.3968  0.3981
#>   0.7492  0.6361
#> [ CPUFloatType{1,4,2,2} ]
#> torch_tensor
#> (1,1,.,.) = 
#>  1.4042  0.5580
#>  -0.6112 -0.0664
#> 
#> (1,2,.,.) = 
#> -0.5515  0.0716
#>  -1.2813 -1.2546
#> 
#> (1,3,.,.) = 
#>  0.6423 -0.1072
#>  -0.0736  0.9928
#> 
#> (1,4,.,.) = 
#>  0.3968  0.3981
#>   0.7492  0.6361
#> [ CPUFloatType{1,4,2,2} ]
```
