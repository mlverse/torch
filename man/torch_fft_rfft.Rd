% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/gen-namespace-docs.R, R/wrapers.R
\name{torch_fft_rfft}
\alias{torch_fft_rfft}
\title{Rfft}
\usage{
torch_fft_rfft(self, n = NULL, dim = -1L, norm = NULL)
}
\arguments{
\item{self}{(Tensor)  the real input tensor}

\item{n}{(int) Signal length. If given, the input will either be zero-padded
or trimmed to this length before computing the real FFT.}

\item{dim}{(int, optional) – The dimension along which to take the one
dimensional real FFT.}

\item{norm}{norm (str, optional) – Normalization mode. For the forward
transform, these correspond to:
\itemize{
\item "forward" - normalize by 1/n
\item "backward" - no normalization
\item "ortho" - normalize by 1/sqrt(n) (making the FFT orthonormal)
Calling the backward transform (\code{\link[=torch_fft_irfft]{torch_fft_irfft()}}) with the same
normalization mode will apply an overall normalization of 1/n between the
two transforms. This is required to make irfft() the exact inverse.
Default is "backward" (no normalization).
}}
}
\description{
Computes the one dimensional Fourier transform of real-valued input.
}
\details{
The FFT of a real signal is Hermitian-symmetric, \code{X[i] = conj(X[-i])} so the
output contains only the positive frequencies below the Nyquist frequency.
To compute the full output, use \code{\link[=torch_fft_fft]{torch_fft_fft()}}.
}
\examples{
if (torch_is_installed()) {
t <- torch_arange(start = 0, end = 3)
torch_fft_rfft(t)

}
}
