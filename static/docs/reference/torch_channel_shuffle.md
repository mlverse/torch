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
#>  1.5760  0.7934
#>  -1.1622 -1.2889
#> 
#> (1,2,.,.) = 
#> -0.7007 -0.3930
#>   1.1782  0.8270
#> 
#> (1,3,.,.) = 
#>  0.6667  0.3059
#>   1.4410  0.6727
#> 
#> (1,4,.,.) = 
#>  0.5557  1.4597
#>   0.0367  0.4386
#> [ CPUFloatType{1,4,2,2} ]
#> torch_tensor
#> (1,1,.,.) = 
#>  1.5760  0.7934
#>  -1.1622 -1.2889
#> 
#> (1,2,.,.) = 
#>  0.6667  0.3059
#>   1.4410  0.6727
#> 
#> (1,3,.,.) = 
#> -0.7007 -0.3930
#>   1.1782  0.8270
#> 
#> (1,4,.,.) = 
#>  0.5557  1.4597
#>   0.0367  0.4386
#> [ CPUFloatType{1,4,2,2} ]
```
