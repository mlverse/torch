# Conv_tbc

Conv_tbc

## Usage

``` r
torch_conv_tbc(self, weight, bias, pad = 0L)
```

## Arguments

- self:

  NA input tensor of shape \\(\mbox{sequence length} \times batch \times
  \mbox{in\\channels})\\

- weight:

  NA filter of shape (\\\mbox{kernel width} \times \mbox{in\\channels}
  \times \mbox{out\\channels}\\)

- bias:

  NA bias of shape (\\\mbox{out\\channels}\\)

- pad:

  NA number of timesteps to pad. Default: 0

## TEST

Applies a 1-dimensional sequence convolution over an input sequence.
Input and output dimensions are (Time, Batch, Channels) - hence TBC.
