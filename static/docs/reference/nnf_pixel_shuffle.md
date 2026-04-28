# Pixel_shuffle

Rearranges elements in a tensor of shape \\(\*, C \times r^2, H, W)\\ to
a tensor of shape \\(\*, C, H \times r, W \times r)\\.

## Usage

``` r
nnf_pixel_shuffle(input, upscale_factor)
```

## Arguments

- input:

  (Tensor) the input tensor

- upscale_factor:

  (int) factor to increase spatial resolution by
