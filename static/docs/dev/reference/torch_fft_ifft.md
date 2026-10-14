# Ifft

Computes the one dimensional inverse discrete Fourier transform of
input.

## Usage

``` r
torch_fft_ifft(self, n = NULL, dim = -1L, norm = NULL)
```

## Arguments

- self:

  (Tensor) the input tensor

- n:

  (int, optional) – Signal length. If given, the input will either be
  zero-padded or trimmed to this length before computing the IFFT.

- dim:

  (int, optional) – The dimension along which to take the one
  dimensional IFFT.

- norm:

  (str, optional) – Normalization mode. For the backward transform,
  these correspond to:

  - "forward" - no normalization

  - "backward" - normalize by 1/n

  - "ortho" - normalize by 1/sqrt(n) (making the IFFT orthonormal)
    Calling the forward transform with the same normalization mode will
    apply an overall normalization of 1/n between the two transforms.
    This is required to make ifft() the exact inverse. Default is
    "backward" (normalize by 1/n).

## Examples

``` r
if (torch_is_installed()) {
t <- torch_arange(start = 0, end = 3)
t
x <- torch_fft_fft(t, norm = "backward")
torch_fft_ifft(x)


}
#> torch_tensor
#> ℹ Use `$real` or `$imag` to print the contents of this tensor.
#> [ CPUComplexFloatType{4} ]
```
