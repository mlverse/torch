# Waits for all kernels in all streams on a CUDA device to complete.

Waits for all kernels in all streams on a CUDA device to complete.

## Usage

``` r
cuda_synchronize(device = NULL)
```

## Arguments

- device:

  device for which to synchronize. It uses the current device given by
  [`cuda_current_device()`](https://torch.mlverse.org/docs/dev/reference/cuda_current_device.md)
  if no device is specified.
