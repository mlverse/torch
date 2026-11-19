# Grid_sample

Given an `input` and a flow-field `grid`, computes the `output` using
`input` values and pixel locations from `grid`.

## Usage

``` r
nnf_grid_sample(
  input,
  grid,
  mode = c("bilinear", "nearest"),
  padding_mode = c("zeros", "border", "reflection"),
  align_corners = FALSE
)
```

## Arguments

- input:

  (Tensor) input of shape \\(N, C, H\_{\mbox{in}}, W\_{\mbox{in}})\\
  (4-D case) or \\(N, C, D\_{\mbox{in}}, H\_{\mbox{in}},
  W\_{\mbox{in}})\\ (5-D case)

- grid:

  (Tensor) flow-field of shape \\(N, H\_{\mbox{out}}, W\_{\mbox{out}},
  2)\\ (4-D case) or \\(N, D\_{\mbox{out}}, H\_{\mbox{out}},
  W\_{\mbox{out}}, 3)\\ (5-D case)

- mode:

  (str) interpolation mode to calculate output values `'bilinear'` \|
  `'nearest'`. Default: `'bilinear'`

- padding_mode:

  (str) padding mode for outside grid values `'zeros'` \| `'border'` \|
  `'reflection'`. Default: `'zeros'`

- align_corners:

  (bool, optional) Geometrically, we consider the pixels of the input as
  squares rather than points. If set to `True`, the extrema (`-1` and
  `1`) are considered as referring to the center points of the input's
  corner pixels. If set to `False`, they are instead considered as
  referring to the corner points of the input's corner pixels, making
  the sampling more resolution agnostic. This option parallels the
  `align_corners` option in
  [`nnf_interpolate()`](https://torch.mlverse.org/docs/dev/reference/nnf_interpolate.md),
  and so whichever option is used here should also be used there to
  resize the input image before grid sampling. Default: `False`

## Details

Currently, only spatial (4-D) and volumetric (5-D) `input` are
supported.

In the spatial (4-D) case, for `input` with shape \\(N, C,
H\_{\mbox{in}}, W\_{\mbox{in}})\\ and `grid` with shape \\(N,
H\_{\mbox{out}}, W\_{\mbox{out}}, 2)\\, the output will have shape \\(N,
C, H\_{\mbox{out}}, W\_{\mbox{out}})\\.

For each output location `output[n, :, h, w]`, the size-2 vector
`grid[n, h, w]` specifies `input` pixel locations `x` and `y`, which are
used to interpolate the output value `output[n, :, h, w]`. In the case
of 5D inputs, `grid[n, d, h, w]` specifies the `x`, `y`, `z` pixel
locations for interpolating `output[n, :, d, h, w]`. `mode` argument
specifies `nearest` or `bilinear` interpolation method to sample the
input pixels.

`grid` specifies the sampling pixel locations normalized by the `input`
spatial dimensions. Therefore, it should have most values in the range
of `[-1, 1]`. For example, values `x = -1, y = -1` is the left-top pixel
of `input`, and values `x = 1, y = 1` is the right-bottom pixel of
`input`.

If `grid` has values outside the range of `[-1, 1]`, the corresponding
outputs are handled as defined by `padding_mode`. Options are

- `padding_mode="zeros"`: use `0` for out-of-bound grid locations,

- `padding_mode="border"`: use border values for out-of-bound grid
  locations,

- `padding_mode="reflection"`: use values at locations reflected by the
  border for out-of-bound grid locations. For location far away from the
  border, it will keep being reflected until becoming in bound, e.g.,
  (normalized) pixel location `x = -3.5` reflects by border `-1` and
  becomes `x' = 1.5`, then reflects by border `1` and becomes
  `x'' = -0.5`.

## Note

This function is often used in conjunction with
[`nnf_affine_grid()`](https://torch.mlverse.org/docs/dev/reference/nnf_affine_grid.md)
to build `Spatial Transformer Networks`\_ .
