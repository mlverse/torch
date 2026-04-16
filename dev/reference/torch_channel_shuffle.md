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
#>  0.9791 -0.9903
#>  -0.6102  1.5586
#> 
#> (1,2,.,.) = 
#>  1.0428 -1.3578
#>   0.4040 -1.9888
#> 
#> (1,3,.,.) = 
#> -0.0751  0.8768
#>  -0.1613  0.3384
#> 
#> (1,4,.,.) = 
#> -1.2934 -0.1999
#>  -0.8737  0.6340
#> [ CPUFloatType{1,4,2,2} ]
#> torch_tensor
#> (1,1,.,.) = 
#>  0.9791 -0.9903
#>  -0.6102  1.5586
#> 
#> (1,2,.,.) = 
#> -0.0751  0.8768
#>  -0.1613  0.3384
#> 
#> (1,3,.,.) = 
#>  1.0428 -1.3578
#>   0.4040 -1.9888
#> 
#> (1,4,.,.) = 
#> -1.2934 -0.1999
#>  -0.8737  0.6340
#> [ CPUFloatType{1,4,2,2} ]
```
