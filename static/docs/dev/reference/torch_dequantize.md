# Dequantize

Dequantize

## Usage

``` r
torch_dequantize(tensor)
```

## Arguments

- tensor:

  (Tensor) A quantized Tensor or a list oof quantized tensors

## dequantize(tensor) -\> Tensor

Returns an fp32 Tensor by dequantizing a quantized Tensor

## dequantize(tensors) -\> sequence of Tensors

Given a list of quantized Tensors, dequantize them and return a list of
fp32 Tensors
