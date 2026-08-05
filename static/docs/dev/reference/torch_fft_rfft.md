# Rfft

Computes the one dimensional Fourier transform of real-valued input.

## Usage

``` r
torch_fft_rfft(self, n = NULL, dim = -1L, norm = NULL)
```

## Arguments

- self:

  (Tensor) the real input tensor

- n:

  (int) Signal length. If given, the input will either be zero-padded or
  trimmed to this length before computing the real FFT.

- dim:

  (int, optional) – The dimension along which to take the one
  dimensional real FFT.

- norm:

  norm (str, optional) – Normalization mode. For the forward transform,
  these correspond to:

  - "forward" - normalize by 1/n

  - "backward" - no normalization

  - "ortho" - normalize by 1/sqrt(n) (making the FFT orthonormal)
    Calling the backward transform
    ([`torch_fft_irfft()`](https://torch.mlverse.org/docs/dev/reference/torch_fft_irfft.md))
    with the same normalization mode will apply an overall normalization
    of 1/n between the two transforms. This is required to make irfft()
    the exact inverse. Default is "backward" (no normalization).

## Details

The FFT of a real signal is Hermitian-symmetric, `X[i] = conj(X[-i])` so
the output contains only the positive frequencies below the Nyquist
frequency. To compute the full output, use
[`torch_fft_fft()`](https://torch.mlverse.org/docs/dev/reference/torch_fft_fft.md).

## Examples

``` r
if (torch_is_installed()) {
t <- torch_arange(start = 0, end = 3)
torch_fft_rfft(t)

}
#> torch_tensor
#> ℹ Use `$real` or `$imag` to print the contents of this tensor.
#> [ CPUComplexFloatType{3} ]
```
