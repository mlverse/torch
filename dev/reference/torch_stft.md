# Stft

Stft

## Usage

``` r
torch_stft(
  input,
  n_fft,
  hop_length = NULL,
  win_length = NULL,
  window = NULL,
  center = TRUE,
  pad_mode = "reflect",
  normalized = FALSE,
  onesided = NULL,
  return_complex = NULL
)
```

## Arguments

- input:

  (Tensor) the input tensor

- n_fft:

  (int) size of Fourier transform

- hop_length:

  (int, optional) the distance between neighboring sliding window
  frames. Default: `NULL` (treated as equal to `floor(n_fft / 4)`)

- win_length:

  (int, optional) the size of window frame and STFT filter. Default:
  `NULL` (treated as equal to `n_fft`)

- window:

  (Tensor, optional) the optional window function. Default: `NULL`
  (treated as window of all \\1\\ s)

- center:

  (bool, optional) whether to pad `input` on both sides so that the
  \\t\\-th frame is centered at time \\t \times \mbox{hop\\length}\\.
  Default: `TRUE`

- pad_mode:

  (string, optional) controls the padding method used when `center` is
  `TRUE`. Default: `"reflect"`

- normalized:

  (bool, optional) controls whether to return the normalized STFT
  results Default: `FALSE`

- onesided:

  (bool, optional) controls whether to return half of results to avoid
  redundancy Default: `TRUE`

- return_complex:

  (bool, optional) controls whether to return complex tensors or not.

## Short-time Fourier transform (STFT).

Short-time Fourier transform (STFT).

    Ignoring the optional batch dimension, this method computes the following
    expression:

\$\$ X\[m, \omega\] = \sum\_{k = 0}^{\mbox{win\\length-1}}%
\mbox{window}\[k\]\\ \mbox{input}\[m \times \mbox{hop\\length} + k\]\\ %
\exp\left(- j \frac{2 \pi \cdot \omega k}{\mbox{win\\length}}\right),
\$\$ where \\m\\ is the index of the sliding window, and \\\omega\\ is
the frequency that \\0 \leq \omega \< \mbox{n\\fft}\\. When `onesided`
is the default value `TRUE`,

    * `input` must be either a 1-D time sequence or a 2-D batch of time
      sequences.

    * If `hop_length` is `NULL` (default), it is treated as equal to
      `floor(n_fft / 4)`.

    * If `win_length` is `NULL` (default), it is treated as equal to
      `n_fft`.

    * `window` can be a 1-D tensor of size `win_length`, e.g., from
      `torch_hann_window`. If `window` is `NULL` (default), it is
      treated as if having \eqn{1} everywhere in the window. If
      \eqn{\mbox{win\_length} < \mbox{n\_fft}}, `window` will be padded on
      both sides to length `n_fft` before being applied.

    * If `center` is `TRUE` (default), `input` will be padded on
      both sides so that the \eqn{t}-th frame is centered at time
      \eqn{t \times \mbox{hop\_length}}. Otherwise, the \eqn{t}-th frame
      begins at time  \eqn{t \times \mbox{hop\_length}}.

    * `pad_mode` determines the padding method used on `input` when
      `center` is `TRUE`. See `torch_nn.functional.pad` for
      all available options. Default is `"reflect"`.

    * If `onesided` is `TRUE` (default), only values for \eqn{\omega}
      in \eqn{\left[0, 1, 2, \dots, \left\lfloor \frac{\mbox{n\_fft}}{2} \right\rfloor + 1\right]}
      are returned because the real-to-complex Fourier transform satisfies the
      conjugate symmetry, i.e., \eqn{X[m, \omega] = X[m, \mbox{n\_fft} - \omega]^*}.

    * If `normalized` is `TRUE` (default is `FALSE`), the function
      returns the normalized STFT results, i.e., multiplied by \eqn{(\mbox{frame\_length})^{-0.5}}.

    Returns the real and the imaginary parts together as one tensor of size
    \eqn{(* \times N \times T \times 2)}, where \eqn{*} is the optional
    batch size of `input`, \eqn{N} is the number of frequencies where
    STFT is applied, \eqn{T} is the total number of frames used, and each pair
    in the last dimension represents a complex number as the real part and the
    imaginary part.

## Warning

This function changed signature at version 0.4.1. Calling with the
previous signature may cause error or return incorrect result.
