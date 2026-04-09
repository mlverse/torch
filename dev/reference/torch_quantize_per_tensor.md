# Quantize_per_tensor

Quantize_per_tensor

## Usage

``` r
torch_quantize_per_tensor(self, scale, zero_point, dtype)
```

## Arguments

- self:

  (Tensor) float tensor to quantize

- scale:

  (float) scale to apply in quantization formula

- zero_point:

  (int) offset in integer value that maps to float zero

- dtype:

  (`torch.dtype`) the desired data type of returned tensor. Has to be
  one of the quantized dtypes: `torch_quint8`, `torch.qint8`,
  `torch.qint32`

## quantize_per_tensor(input, scale, zero_point, dtype) -\> Tensor

Converts a float tensor to quantized tensor with given scale and zero
point.

## Examples

``` r
if (torch_is_installed()) {
torch_quantize_per_tensor(torch_tensor(c(-1.0, 0.0, 1.0, 2.0)), 0.1, 10, torch_quint8())
torch_quantize_per_tensor(torch_tensor(c(-1.0, 0.0, 1.0, 2.0)), 0.1, 10, torch_quint8())$int_repr()
}
#> torch_tensor
#>   0
#>  10
#>  20
#>  30
#> [ CPUByteType{4} ]
```
