% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/creation-ops.R, R/gen-namespace-docs.R,
%   R/gen-namespace-examples.R
\name{torch_randperm}
\alias{torch_randperm}
\title{Randperm}
\usage{
torch_randperm(
  n,
  dtype = torch_int64(),
  layout = NULL,
  device = NULL,
  requires_grad = FALSE
)
}
\arguments{
\item{n}{(int) the upper bound (exclusive)}

\item{dtype}{(\code{torch.dtype}, optional) the desired data type of returned tensor.        Default: \code{torch_int64}.}

\item{layout}{(\code{torch.layout}, optional) the desired layout of returned Tensor.        Default: \code{torch_strided}.}

\item{device}{(\code{torch.device}, optional) the desired device of returned tensor.        Default: if \code{NULL}, uses the current device for the default tensor type        (see \code{torch_set_default_tensor_type}). \code{device} will be the CPU        for CPU tensor types and the current CUDA device for CUDA tensor types.}

\item{requires_grad}{(bool, optional) If autograd should record operations on the        returned tensor. Default: \code{FALSE}.}
}
\description{
Randperm
}
\section{randperm(n, out=NULL, dtype=torch.int64, layout=torch.strided, device=NULL, requires_grad=False) -> LongTensor }{


Returns a random permutation of integers from \code{0} to \code{n - 1}.
}

\examples{
if (torch_is_installed()) {

torch_randperm(4)
}
}
