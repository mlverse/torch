# Quantize_per_channel

Quantize_per_channel

## Usage

``` r
torch_quantize_per_channel(self, scales, zero_points, axis, dtype)
```

## Arguments

- self:

  (Tensor) float tensor to quantize

- scales:

  (Tensor) float 1D tensor of scales to use, size should match
  `input.size(axis)`

- zero_points:

  (int) integer 1D tensor of offset to use, size should match
  `input.size(axis)`

- axis:

  (int) dimension on which apply per-channel quantization

- dtype:

  (`torch.dtype`) the desired data type of returned tensor. Has to be
  one of the quantized dtypes: `torch_quint8`, `torch.qint8`,
  `torch.qint32`

## quantize_per_channel(input, scales, zero_points, axis, dtype) -\> Tensor

Converts a float tensor to per-channel quantized tensor with given
scales and zero points.

## Examples

``` r
if (torch_is_installed()) {
x = torch_tensor(matrix(c(-1.0, 0.0, 1.0, 2.0), ncol = 2, byrow = TRUE))
torch_quantize_per_channel(x, torch_tensor(c(0.1, 0.01)), 
                           torch_tensor(c(10L, 0L)), 0, torch_quint8())
torch_quantize_per_channel(x, torch_tensor(c(0.1, 0.01)), 
                           torch_tensor(c(10L, 0L)), 0, torch_quint8())$int_repr()
}
#> torch_tensor
#>    0   10
#>  100  200
#> [ CPUByteType{2,2} ]
```
