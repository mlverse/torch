% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/gen-namespace-docs.R,
%   R/gen-namespace-examples.R, R/wrapers.R
\name{torch_stft}
\alias{torch_stft}
\title{Stft}
\usage{
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
}
\arguments{
\item{input}{(Tensor) the input tensor}

\item{n_fft}{(int) size of Fourier transform}

\item{hop_length}{(int, optional) the distance between neighboring sliding window
frames. Default: \code{NULL} (treated as equal to \code{floor(n_fft / 4)})}

\item{win_length}{(int, optional) the size of window frame and STFT filter.
Default: \code{NULL}  (treated as equal to \code{n_fft})}

\item{window}{(Tensor, optional) the optional window function.
Default: \code{NULL} (treated as window of all \eqn{1} s)}

\item{center}{(bool, optional) whether to pad \code{input} on both sides so
that the \eqn{t}-th frame is centered at time \eqn{t \times \mbox{hop\_length}}.
Default: \code{TRUE}}

\item{pad_mode}{(string, optional) controls the padding method used when
\code{center} is \code{TRUE}. Default: \code{"reflect"}}

\item{normalized}{(bool, optional) controls whether to return the normalized
STFT results Default: \code{FALSE}}

\item{onesided}{(bool, optional) controls whether to return half of results to
avoid redundancy Default: \code{TRUE}}

\item{return_complex}{(bool, optional) controls whether to return complex tensors
or not.}
}
\description{
Stft
}
\section{Short-time Fourier transform (STFT). }{


Short-time Fourier transform (STFT).

\if{html}{\out{<div class="sourceCode">}}\preformatted{Ignoring the optional batch dimension, this method computes the following
expression:
}\if{html}{\out{</div>}}

\deqn{
        X[m, \omega] = \sum_{k = 0}^{\mbox{win\_length-1}}%
                            \mbox{window}[k]\ \mbox{input}[m \times \mbox{hop\_length} + k]\ %
                            \exp\left(- j \frac{2 \pi \cdot \omega k}{\mbox{win\_length}}\right),
}
where \eqn{m} is the index of the sliding window, and \eqn{\omega} is
the frequency that \eqn{0 \leq \omega < \mbox{n\_fft}}. When
\code{onesided} is the default value \code{TRUE},

\if{html}{\out{<div class="sourceCode">}}\preformatted{* `input` must be either a 1-D time sequence or a 2-D batch of time
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
}\if{html}{\out{</div>}}
}

\section{Warning}{

This function changed signature at version 0.4.1. Calling with the
previous signature may cause error or return incorrect result.
}

