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
#> -0.9076 -0.4541
#>   1.3999 -1.7503
#> 
#> (1,2,.,.) = 
#>  0.0515  0.3161
#>  -0.2851  0.5931
#> 
#> (1,3,.,.) = 
#> -0.5520  1.1620
#>  -0.9336 -1.4470
#> 
#> (1,4,.,.) = 
#> -0.8159 -0.3717
#>  -0.3645  0.0862
#> [ CPUFloatType{1,4,2,2} ]
#> torch_tensor
#> (1,1,.,.) = 
#> -0.9076 -0.4541
#>   1.3999 -1.7503
#> 
#> (1,2,.,.) = 
#> -0.5520  1.1620
#>  -0.9336 -1.4470
#> 
#> (1,3,.,.) = 
#>  0.0515  0.3161
#>  -0.2851  0.5931
#> 
#> (1,4,.,.) = 
#> -0.8159 -0.3717
#>  -0.3645  0.0862
#> [ CPUFloatType{1,4,2,2} ]
```
