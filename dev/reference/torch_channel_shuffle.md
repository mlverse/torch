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
#>  0.01 *
#>  -7.1518 -0.2590
#>   -11.6317 -173.1409
#> 
#> (1,2,.,.) = 
#>   2.0135 -1.2041
#>  -0.5985 -1.1337
#> 
#> (1,3,.,.) = 
#>   0.9751 -0.5116
#>  -1.1883  0.9457
#> 
#> (1,4,.,.) = 
#>   1.9780  1.4006
#>  -0.2872  1.5418
#> [ CPUFloatType{1,4,2,2} ]
#> torch_tensor
#> (1,1,.,.) = 
#>  0.01 *
#>  -7.1518 -0.2590
#>   -11.6317 -173.1409
#> 
#> (1,2,.,.) = 
#>   0.9751 -0.5116
#>  -1.1883  0.9457
#> 
#> (1,3,.,.) = 
#>   2.0135 -1.2041
#>  -0.5985 -1.1337
#> 
#> (1,4,.,.) = 
#>   1.9780  1.4006
#>  -0.2872  1.5418
#> [ CPUFloatType{1,4,2,2} ]
```
