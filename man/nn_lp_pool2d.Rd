% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/nn-pooling.R
\name{nn_lp_pool2d}
\alias{nn_lp_pool2d}
\title{Applies a 2D power-average pooling over an input signal composed of several input
planes.}
\usage{
nn_lp_pool2d(norm_type, kernel_size, stride = NULL, ceil_mode = FALSE)
}
\arguments{
\item{norm_type}{if inf than one gets max pooling if 0 you get sum pooling (
proportional to the avg pooling)}

\item{kernel_size}{the size of the window}

\item{stride}{the stride of the window. Default value is \code{kernel_size}}

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
\item At p = 1, one gets Sum Pooling (which is proportional to average pooling)
}

The parameters \code{kernel_size}, \code{stride} can either be:
\itemize{
\item a single \code{int} -- in which case the same value is used for the height and width dimension
\item a \code{tuple} of two ints -- in which case, the first \code{int} is used for the height dimension,
and the second \code{int} for the width dimension
}
}
\note{
If the sum to the power of \code{p} is zero, the gradient of this function is
not defined. This implementation will set the gradient to zero in this case.
}
\section{Shape}{

\itemize{
\item Input: \eqn{(N, C, H_{in}, W_{in})}
\item Output: \eqn{(N, C, H_{out}, W_{out})}, where
}

\deqn{
  H_{out} = \left\lfloor\frac{H_{in} - \mbox{kernel\_size}[0]}{\mbox{stride}[0]} + 1\right\rfloor
}
\deqn{
  W_{out} = \left\lfloor\frac{W_{in} - \mbox{kernel\_size}[1]}{\mbox{stride}[1]} + 1\right\rfloor
}
}

\examples{
if (torch_is_installed()) {

# power-2 pool of square window of size=3, stride=2
m <- nn_lp_pool2d(2, 3, stride = 2)
# pool of non-square window of power 1.2
m <- nn_lp_pool2d(1.2, c(3, 2), stride = c(2, 1))
input <- torch_randn(20, 16, 50, 32)
output <- m(input)
}
}
