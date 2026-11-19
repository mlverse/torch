# Conv_tbc

Applies a 1-dimensional sequence convolution over an input sequence.
Input and output dimensions are (Time, Batch, Channels) - hence TBC.

## Usage

``` r
nnf_conv_tbc(input, weight, bias, pad = 0)
```

## Arguments

- input:

  input tensor of shape \\(\mbox{sequence length} \times batch \times
  \mbox{in\\channels})\\

- weight:

  filter of shape (\\\mbox{kernel width} \times \mbox{in\\channels}
  \times \mbox{out\\channels}\\)

- bias:

  bias of shape (\\\mbox{out\\channels}\\)

- pad:

  number of timesteps to pad. Default: 0
