# Interpolate

Down/up samples the input to either the given `size` or the given
`scale_factor`

## Usage

``` r
nnf_interpolate(
  input,
  size = NULL,
  scale_factor = NULL,
  mode = "nearest",
  align_corners = FALSE,
  recompute_scale_factor = NULL
)
```

## Arguments

- input:

  (Tensor) the input tensor

- size:

  (int or `Tuple[int]` or `Tuple[int, int]` or `Tuple[int, int, int]`)
  output spatial size.

- scale_factor:

  (float or `Tuple[float]`) multiplier for spatial size. Has to match
  input size if it is a tuple.

- mode:

  (str) algorithm used for upsampling: 'nearest' \| 'linear' \|
  'bilinear' \| 'bicubic' \| 'trilinear' \| 'area' Default: 'nearest'

- align_corners:

  (bool, optional) Geometrically, we consider the pixels of the input
  and output as squares rather than points. If set to TRUE, the input
  and output tensors are aligned by the center points of their corner
  pixels, preserving the values at the corner pixels. If set to False,
  the input and output tensors are aligned by the corner points of their
  corner pixels, and the interpolation uses edge value padding for
  out-of-boundary values, making this operation *independent* of input
  size when `scale_factor` is kept the same. This only has an effect
  when `mode` is `'linear'`, `'bilinear'`, `'bicubic'` or `'trilinear'`.
  Default: `False`

- recompute_scale_factor:

  (bool, optional) recompute the scale_factor for use in the
  interpolation calculation. When `scale_factor` is passed as a
  parameter, it is used to compute the `output_size`. If
  `recompute_scale_factor` is “\`True“ or not specified, a new
  `scale_factor` will be computed based on the output and input sizes
  for use in the interpolation computation (i.e. the computation will be
  identical to if the computed \`output_size\` were passed-in
  explicitly). Otherwise, the passed-in \`scale_factor\` will be used in
  the interpolation computation. Note that when \`scale_factor\` is
  floating-point, the recomputed scale_factor may differ from the one
  passed in due to rounding and precision issues.

## Details

The algorithm used for interpolation is determined by `mode`.

Currently temporal, spatial and volumetric sampling are supported, i.e.
expected inputs are 3-D, 4-D or 5-D in shape.

The input dimensions are interpreted in the form:
`mini-batch x channels x [optional depth] x [optional height] x width`.

The modes available for resizing are: `nearest`, `linear` (3D-only),
`bilinear`, `bicubic` (4D-only), `trilinear` (5D-only), `area`
