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
#>  -7.7935 -51.9032
#>   -53.5792 -94.5352
#> 
#> (1,2,.,.) = 
#>  -0.3270  0.4718
#>  -1.4228  0.4581
#> 
#> (1,3,.,.) = 
#>  -0.0720  0.2146
#>   1.5004  0.2347
#> 
#> (1,4,.,.) = 
#>  -0.6492 -0.9518
#>  -2.0361 -1.1655
#> [ CPUFloatType{1,4,2,2} ]
#> torch_tensor
#> (1,1,.,.) = 
#>  0.01 *
#>  -7.7935 -51.9032
#>   -53.5792 -94.5352
#> 
#> (1,2,.,.) = 
#>  -0.0720  0.2146
#>   1.5004  0.2347
#> 
#> (1,3,.,.) = 
#>  -0.3270  0.4718
#>  -1.4228  0.4581
#> 
#> (1,4,.,.) = 
#>  -0.6492 -0.9518
#>  -2.0361 -1.1655
#> [ CPUFloatType{1,4,2,2} ]
```
