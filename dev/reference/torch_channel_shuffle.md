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
#>  1.1639 -0.4308
#>  -0.5171 -1.5302
#> 
#> (1,2,.,.) = 
#> -2.1431 -0.1137
#>   0.6292  0.9542
#> 
#> (1,3,.,.) = 
#>  0.4528  0.2609
#>  -0.1189 -0.3731
#> 
#> (1,4,.,.) = 
#>  0.4995  1.3780
#>   0.9910  1.9573
#> [ CPUFloatType{1,4,2,2} ]
#> torch_tensor
#> (1,1,.,.) = 
#>  1.1639 -0.4308
#>  -0.5171 -1.5302
#> 
#> (1,2,.,.) = 
#>  0.4528  0.2609
#>  -0.1189 -0.3731
#> 
#> (1,3,.,.) = 
#> -2.1431 -0.1137
#>   0.6292  0.9542
#> 
#> (1,4,.,.) = 
#>  0.4995  1.3780
#>   0.9910  1.9573
#> [ CPUFloatType{1,4,2,2} ]
```
