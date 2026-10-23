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
#> -0.7959 -0.9737
#>   0.3114 -0.0758
#> 
#> (1,2,.,.) = 
#>  0.6239 -0.3699
#>   0.3358 -0.2103
#> 
#> (1,3,.,.) = 
#> -0.0042 -0.6144
#>  -1.3993 -1.0631
#> 
#> (1,4,.,.) = 
#> -3.3195  0.7471
#>   0.0773  1.3314
#> [ CPUFloatType{1,4,2,2} ]
#> torch_tensor
#> (1,1,.,.) = 
#> -0.7959 -0.9737
#>   0.3114 -0.0758
#> 
#> (1,2,.,.) = 
#> -0.0042 -0.6144
#>  -1.3993 -1.0631
#> 
#> (1,3,.,.) = 
#>  0.6239 -0.3699
#>   0.3358 -0.2103
#> 
#> (1,4,.,.) = 
#> -3.3195  0.7471
#>   0.0773  1.3314
#> [ CPUFloatType{1,4,2,2} ]
```
