# Pad

Pads tensor.

## Usage

``` r
nnf_pad(input, pad, mode = "constant", value = NULL)
```

## Arguments

- input:

  (Tensor) N-dimensional tensor

- pad:

  (tuple) m-elements tuple, where \\\frac{m}{2} \leq\\ input dimensions
  and \\m\\ is even.

- mode:

  'constant', 'reflect', 'replicate' or 'circular'. Default: 'constant'

- value:

  fill value for 'constant' padding. Default: 0.

## Padding size

The padding size by which to pad some dimensions of `input` are
described starting from the last dimension and moving forward.
\\\left\lfloor\frac{\mbox{len(pad)}}{2}\right\rfloor\\ dimensions of
`input` will be padded. For example, to pad only the last dimension of
the input tensor, then `pad` has the form \\(\mbox{padding\\left},
\mbox{padding\\right})\\; to pad the last 2 dimensions of the input
tensor, then use \\(\mbox{padding\\left}, \mbox{padding\\right},\\
\\\mbox{padding\\top}, \mbox{padding\\bottom})\\; to pad the last 3
dimensions, use \\(\mbox{padding\\left}, \mbox{padding\\right},\\
\\\mbox{padding\\top}, \mbox{padding\\bottom}\\ \\\mbox{padding\\front},
\mbox{padding\\back})\\.

## Padding mode

See `nn_constant_pad_2d`, `nn_reflection_pad_2d`, and
`nn_replication_pad_2d` for concrete examples on how each of the padding
modes works. Constant padding is implemented for arbitrary dimensions.
tensor, or the last 2 dimensions of 4D input tensor, or the last
dimension of 3D input tensor. Reflect padding is only implemented for
padding the last 2 dimensions of 4D input tensor, or the last dimension
of 3D input tensor.
