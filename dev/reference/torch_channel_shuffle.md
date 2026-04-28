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
#>  1.3379 -0.4217
#>  -0.2014 -0.3484
#> 
#> (1,2,.,.) = 
#> -0.9286  1.9177
#>   0.7558  0.4857
#> 
#> (1,3,.,.) = 
#>  0.7517  0.8833
#>  -1.0322  0.3357
#> 
#> (1,4,.,.) = 
#> -0.7671  1.2872
#>   0.6578  0.3416
#> [ CPUFloatType{1,4,2,2} ]
#> torch_tensor
#> (1,1,.,.) = 
#>  1.3379 -0.4217
#>  -0.2014 -0.3484
#> 
#> (1,2,.,.) = 
#>  0.7517  0.8833
#>  -1.0322  0.3357
#> 
#> (1,3,.,.) = 
#> -0.9286  1.9177
#>   0.7558  0.4857
#> 
#> (1,4,.,.) = 
#> -0.7671  1.2872
#>   0.6578  0.3416
#> [ CPUFloatType{1,4,2,2} ]
```
