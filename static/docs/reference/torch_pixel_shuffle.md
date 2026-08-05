# Pixel_shuffle

Pixel_shuffle

## Usage

``` r
torch_pixel_shuffle(self, upscale_factor)
```

## Arguments

- self:

  (Tensor) the input tensor

- upscale_factor:

  (int) factor to increase spatial resolution by

## Rearranges elements in a tensor of shape

math:`(*, C \times r^2, H, W)` to a :

Rearranges elements in a tensor of shape \\(\*, C \times r^2, H, W)\\ to
a tensor of shape \\(\*, C, H \times r, W \times r)\\.

See `~torch.nn.PixelShuffle` for details.

## Examples

``` r
if (torch_is_installed()) {

input = torch_randn(c(1, 9, 4, 4))
output = nnf_pixel_shuffle(input, 3)
print(output$size())
}
#> [1]  1  1 12 12
```
