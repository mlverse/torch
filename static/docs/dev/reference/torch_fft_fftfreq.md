# fftfreq

Computes the discrete Fourier Transform sample frequencies for a signal
of size `n`.

## Usage

``` r
torch_fft_fftfreq(
  n,
  d = 1,
  dtype = NULL,
  layout = NULL,
  device = NULL,
  requires_grad = FALSE
)
```

## Arguments

- n:

  (integer) – the FFT length

- d:

  (float, optional) – the sampling length scale. The spacing between
  individual samples of the FFT input. The default assumes unit spacing,
  dividing that result by the actual spacing gives the result in
  physical frequency units.

- dtype:

  (default:
  [`torch_get_default_dtype()`](https://torch.mlverse.org/docs/dev/reference/default_dtype.md))
  the desired data type of returned tensor.

- layout:

  (default:
  [`torch_strided()`](https://torch.mlverse.org/docs/dev/reference/torch_layout.md))
  the desired layout of returned tensor.

- device:

  (default: `NULL`) the desired device of returned tensor. Default: If
  `NULL`, uses the current device for the default tensor type.

- requires_grad:

  (default: `FALSE`) If autograd should record operations on the
  returned tensor.

## Note

By convention,
[`torch_fft_fft()`](https://torch.mlverse.org/docs/dev/reference/torch_fft_fft.md)
returns positive frequency terms first, followed by the negative
frequencies in reverse order, so that `f[-i]` for all `0 < i <= n/2`
gives the negative frequency terms. For an FFT of length `n` and with
inputs spaced in length unit `d`, the frequencies are:
`f = [0, 1, ..., (n - 1) // 2, -(n // 2), ..., -1] / (d * n)`

For even lengths, the Nyquist frequency at `f[n/2]` can be thought of as
either negative or positive. `fftfreq()` follows NumPy’s convention of
taking it to be negative.

## Examples

``` r
if (torch_is_installed()) {
torch_fft_fftfreq(5) # Nyquist frequency at f[3] is positive
torch_fft_fftfreq(4) # Nyquist frequency at f[3] is given as negative

}
#> torch_tensor
#>  0.0000
#>  0.2500
#> -0.5000
#> -0.2500
#> [ CPUFloatType{4} ]
```
