# Upsample module

Upsamples a given multi-channel 1D (temporal), 2D (spatial) or 3D
(volumetric) data. The input data is assumed to be of the form minibatch
x channels x optional depth x optional height\] x width. Hence, for
spatial inputs, we expect a 4D Tensor and for volumetric inputs, we
expect a 5D Tensor.

## Usage

``` r
nn_upsample(
  size = NULL,
  scale_factor = NULL,
  mode = "nearest",
  align_corners = NULL
)
```

## Arguments

- size:

  (int or `Tuple[int]` or `Tuple[int, int]` or `Tuple[int, int, int]`,
  optional): output spatial sizes

- scale_factor:

  (float or `Tuple[float]` or `Tuple[float, float]` or
  `Tuple[float, float, float]`, optional): multiplier for spatial size.
  Has to match input size if it is a tuple.

- mode:

  (str, optional): the upsampling algorithm: one of `'nearest'`,
  `'linear'`, `'bilinear'`, `'bicubic'` and `'trilinear'`. Default:
  `'nearest'`

- align_corners:

  (bool, optional): if `TRUE`, the corner pixels of the input and output
  tensors are aligned, and thus preserving the values at those pixels.
  This only has effect when `mode` is `'linear'`, `'bilinear'`, or
  `'trilinear'`. Default: `FALSE`

## Details

The algorithms available for upsampling are nearest neighbor and linear,
bilinear, bicubic and trilinear for 3D, 4D and 5D input Tensor,
respectively.

One can either give a scale_factor or the target output size to
calculate the output size. (You cannot give both, as it is ambiguous)

## Examples

``` r
if (torch_is_installed()) {
input <- torch_arange(start = 1, end = 4, dtype = torch_float())$view(c(1, 1, 2, 2))
nn_upsample(scale_factor = c(2), mode = "nearest")(input)
nn_upsample(scale_factor = c(2, 2), mode = "nearest")(input)
}
#> torch_tensor
#> (1,1,.,.) = 
#>   1  1  2  2
#>   1  1  2  2
#>   3  3  4  4
#>   3  3  4  4
#> [ CPUFloatType{1,1,4,4} ]
```
