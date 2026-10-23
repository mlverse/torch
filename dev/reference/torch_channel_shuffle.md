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
#>  1.1376  1.0635
#>   0.0679  0.5173
#> 
#> (1,2,.,.) = 
#> -1.4211 -1.7723
#>   1.1434 -0.3125
#> 
#> (1,3,.,.) = 
#>  0.9969 -2.4550
#>  -1.9617 -0.5171
#> 
#> (1,4,.,.) = 
#> -0.3016 -0.1357
#>   3.2738 -1.0909
#> [ CPUFloatType{1,4,2,2} ]
#> torch_tensor
#> (1,1,.,.) = 
#>  1.1376  1.0635
#>   0.0679  0.5173
#> 
#> (1,2,.,.) = 
#>  0.9969 -2.4550
#>  -1.9617 -0.5171
#> 
#> (1,3,.,.) = 
#> -1.4211 -1.7723
#>   1.1434 -0.3125
#> 
#> (1,4,.,.) = 
#> -0.3016 -0.1357
#>   3.2738 -1.0909
#> [ CPUFloatType{1,4,2,2} ]
```
