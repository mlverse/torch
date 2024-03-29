% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/nn-pooling.R
\name{nn_lp_pool1d}
\alias{nn_lp_pool1d}
\title{Applies a 1D power-average pooling over an input signal composed of several input
planes.}
\usage{
nn_lp_pool1d(norm_type, kernel_size, stride = NULL, ceil_mode = FALSE)
}
\arguments{
\item{norm_type}{if inf than one gets max pooling if 0 you get sum pooling (
proportional to the avg pooling)}

\item{kernel_size}{a single int, the size of the window}

\item{stride}{a single int, the stride of the window. Default value is \code{kernel_size}}

\item{ceil_mode}{when TRUE, will use \code{ceil} instead of \code{floor} to compute the output shape}
}
\description{
On each window, the function computed is:
}
\details{
\deqn{
  f(X) = \sqrt[p]{\sum_{x \in X} x^{p}}
}
\itemize{
\item At p = \eqn{\infty}, one gets Max Pooling
\item At p = 1, one gets Sum Pooling (which is proportional to Average Pooling)
}
}
\note{
If the sum to the power of \code{p} is zero, the gradient of this function is
not defined. This implementation will set the gradient to zero in this case.
}
\section{Shape}{

\itemize{
\item Input: \eqn{(N, C, L_{in})}
\item Output: \eqn{(N, C, L_{out})}, where
}

\deqn{
  L_{out} = \left\lfloor\frac{L_{in} - \mbox{kernel\_size}}{\mbox{stride}} + 1\right\rfloor
}
}

\examples{
if (torch_is_installed()) {
# power-2 pool of window of length 3, with stride 2.
m <- nn_lp_pool1d(2, 3, stride = 2)
input <- torch_randn(20, 16, 50)
output <- m(input)
}
}
