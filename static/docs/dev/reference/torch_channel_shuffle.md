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
#>   2.2912 -2.6283
#>   1.5552  0.3406
#> 
#> (1,2,.,.) = 
#>  -0.6944  0.1834
#>  -1.9672  1.8651
#> 
#> (1,3,.,.) = 
#>   1.7977 -0.4962
#>  -0.6441  0.4838
#> 
#> (1,4,.,.) = 
#>  -0.3788 -0.2935
#>   0.2241  0.7408
#> [ CPUFloatType{1,4,2,2} ]
#> torch_tensor
#> (1,1,.,.) = 
#>   2.2912 -2.6283
#>   1.5552  0.3406
#> 
#> (1,2,.,.) = 
#>   1.7977 -0.4962
#>  -0.6441  0.4838
#> 
#> (1,3,.,.) = 
#>  -0.6944  0.1834
#>  -1.9672  1.8651
#> 
#> (1,4,.,.) = 
#>  -0.3788 -0.2935
#>   0.2241  0.7408
#> [ CPUFloatType{1,4,2,2} ]
```
