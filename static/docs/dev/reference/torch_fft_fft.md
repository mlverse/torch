# Fft

Computes the one dimensional discrete Fourier transform of input.

## Usage

``` r
torch_fft_fft(self, n = NULL, dim = -1L, norm = NULL)
```

## Arguments

- self:

  (Tensor) the input tensor

- n:

  (int) Signal length. If given, the input will either be zero-padded or
  trimmed to this length before computing the FFT.

- dim:

  (int, optional) The dimension along which to take the one dimensional
  FFT.

- norm:

  (str, optional) Normalization mode. For the forward transform, these
  correspond to:

  - "forward" - normalize by 1/n

  - "backward" - no normalization

  - "ortho" - normalize by 1/sqrt(n) (making the FFT orthonormal)
    Calling the backward transform (ifft()) with the same normalization
    mode will apply an overall normalization of 1/n between the two
    transforms. This is required to make IFFT the exact inverse. Default
    is "backward" (no normalization).

## Note

The Fourier domain representation of any real signal satisfies the
Hermitian property: `X[i] = conj(X[-i]).` This function always returns
both the positive and negative frequency terms even though, for real
inputs, the negative frequencies are redundant. rfft() returns the more
compact one-sided representation where only the positive frequencies are
returned.

## Examples

``` r
if (torch_is_installed()) {
t <- torch_arange(start = 0, end = 3)
t
torch_fft_fft(t, norm = "backward")

}
#> torch_tensor
#> ℹ Use `$real` or `$imag` to print the contents of this tensor.
#> [ CPUComplexFloatType{4} ]
```
