% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/gen-namespace-docs.R,
%   R/gen-namespace-examples.R, R/gen-namespace.R
\name{torch_quantile}
\alias{torch_quantile}
\title{Quantile}
\usage{
torch_quantile(self, q, dim = NULL, keepdim = FALSE, interpolation = "linear")
}
\arguments{
\item{self}{(Tensor) the input tensor.}

\item{q}{(float or Tensor) a scalar or 1D tensor of quantile values in the range \verb{[0, 1]}}

\item{dim}{(int) the dimension to reduce.}

\item{keepdim}{(bool) whether the output tensor has \code{dim} retained or not.}

\item{interpolation}{The interpolation method.}
}
\description{
Quantile
}
\section{quantile(input, q) -> Tensor }{


Returns the q-th quantiles of all elements in the \code{input} tensor, doing a linear
interpolation when the q-th quantile lies between two data points.
}

\section{quantile(input, q, dim=None, keepdim=FALSE, *, out=None) -> Tensor }{


Returns the q-th quantiles of each row of the \code{input} tensor along the dimension
\code{dim}, doing a linear interpolation when the q-th quantile lies between two
data points. By default, \code{dim} is \code{None} resulting in the \code{input} tensor
being flattened before computation.

If \code{keepdim} is \code{TRUE}, the output dimensions are of the same size as \code{input}
except in the dimensions being reduced (\code{dim} or all if \code{dim} is \code{NULL}) where they
have size 1. Otherwise, the dimensions being reduced are squeezed (see \code{\link{torch_squeeze}}).
If \code{q} is a 1D tensor, an extra dimension is prepended to the output tensor with the same
size as \code{q} which represents the quantiles.
}

\examples{
if (torch_is_installed()) {

a <- torch_randn(c(1, 3))
a
q <- torch_tensor(c(0, 0.5, 1))
torch_quantile(a, q)


a <- torch_randn(c(2, 3))
a
q <- torch_tensor(c(0.25, 0.5, 0.75))
torch_quantile(a, q, dim=1, keepdim=TRUE)
torch_quantile(a, q, dim=1, keepdim=TRUE)$shape
}
}
