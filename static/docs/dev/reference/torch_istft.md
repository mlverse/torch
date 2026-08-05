# Istft

Inverse short time Fourier Transform. This is expected to be the inverse
of
[`torch_stft()`](https://torch.mlverse.org/docs/dev/reference/torch_stft.md).

## Usage

``` r
torch_istft(
  self,
  n_fft,
  hop_length = NULL,
  win_length = NULL,
  window = list(),
  center = TRUE,
  normalized = FALSE,
  onesided = NULL,
  length = NULL,
  return_complex = FALSE
)
```

## Arguments

- self:

  (Tensor) The input tensor. Expected to be output of
  [`torch_stft()`](https://torch.mlverse.org/docs/dev/reference/torch_stft.md),
  can either be complex (`channel`, `fft_size`, `n_frame`), or real
  (`channel`, `fft_size`, `n_frame`, 2) where the `channel` dimension is
  optional.

- n_fft:

  (int) Size of Fourier transform

- hop_length:

  (Optional`[int]`) The distance between neighboring sliding window
  frames. (Default: `n_fft %% 4`)

- win_length:

  (Optional`[int]`) The size of window frame and STFT filter. (Default:
  `n_fft`)

- window:

  (Optional(torch.Tensor)) The optional window function. (Default:
  `torch_ones(win_length)`)

- center:

  (bool) Whether `input` was padded on both sides so that the \\t\\-th
  frame is centered at time \\t \times \mbox{hop\\length}\\. (Default:
  `TRUE`)

- normalized:

  (bool) Whether the STFT was normalized. (Default: `FALSE`)

- onesided:

  (Optional(bool)) Whether the STFT was onesided. (Default: `TRUE` if
  `n_fft != fft_size` in the input size)

- length:

  (Optional(int)\]) The amount to trim the signal by (i.e. the original
  signal length). (Default: whole signal)

- return_complex:

  (Optional(bool)) Whether the output should be complex, or if the input
  should be assumed to derive from a real signal and window. Note that
  this is incompatible with `onesided=TRUE`. (Default: `FALSE`)

## Details

It has the same parameters (+ additional optional parameter of `length`)
and it should return the least squares estimation of the original
signal. The algorithm will check using the NOLA condition (nonzero
overlap).

Important consideration in the parameters `window` and `center` so that
the envelop created by the summation of all the windows is never zero at
certain point in time. Specifically, \\\sum\_{t=-\infty}^{\infty}
\|w\|^2(n-t\times hop_length) \neq 0\\.

Since
[`torch_stft()`](https://torch.mlverse.org/docs/dev/reference/torch_stft.md)
discards elements at the end of the signal if they do not fit in a
frame, `istft` may return a shorter signal than the original signal (can
occur if `center` is FALSE since the signal isn't padded).

If `center` is `TRUE`, then there will be padding e.g. `'constant'`,
`'reflect'`, etc. Left padding can be trimmed off exactly because they
can be calculated but right padding cannot be calculated without
additional information.

Example: Suppose the last window is: `[c(17, 18, 0, 0, 0)` vs
`c(18, 0, 0, 0, 0)`

The `n_fft`, `hop_length`, `win_length` are all the same which prevents
the calculation of right padding. These additional values could be zeros
or a reflection of the signal so providing `length` could be useful. If
`length` is `None` then padding will be aggressively removed (some loss
of signal).

D. W. Griffin and J. S. Lim, "Signal estimation from modified short-time
Fourier transform," IEEE Trans. ASSP, vol.32, no.2, pp.236-243, Apr.
1984.
