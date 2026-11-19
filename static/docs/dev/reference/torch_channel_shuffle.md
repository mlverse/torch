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
#> -0.8866  0.8048
#>   0.1130 -0.8799
#> 
#> (1,2,.,.) = 
#> -0.1828  1.4054
#>  -0.4523 -0.8444
#> 
#> (1,3,.,.) = 
#>  1.1789  1.3649
#>   0.6730  0.8615
#> 
#> (1,4,.,.) = 
#> -2.1032 -0.0048
#>  -1.5002  0.4248
#> [ CPUFloatType{1,4,2,2} ]
#> torch_tensor
#> (1,1,.,.) = 
#> -0.8866  0.8048
#>   0.1130 -0.8799
#> 
#> (1,2,.,.) = 
#>  1.1789  1.3649
#>   0.6730  0.8615
#> 
#> (1,3,.,.) = 
#> -0.1828  1.4054
#>  -0.4523 -0.8444
#> 
#> (1,4,.,.) = 
#> -2.1032 -0.0048
#>  -1.5002  0.4248
#> [ CPUFloatType{1,4,2,2} ]
```
