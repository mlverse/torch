% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/nnf-activation.R
\name{nnf_gelu}
\alias{nnf_gelu}
\title{Gelu}
\usage{
nnf_gelu(input, approximate = "none")
}
\arguments{
\item{input}{(N,*) tensor, where * means, any number of additional
dimensions}

\item{approximate}{By default it's none, and applies element-wise x*pnorm(x),
if 'tanh', then GELU is estimated. See \href{https://arxiv.org/abs/1606.08415}{GELU} for
more info.}
}
\description{
Gelu
}
\section{gelu(input) -> Tensor }{


Applies element-wise the function
\eqn{GELU(x) = x * \Phi(x)}

where \eqn{\Phi(x)} is the Cumulative Distribution Function for
Gaussian Distribution.

See \href{https://arxiv.org/abs/1606.08415}{Gaussian Error Linear Units (GELUs)}.
}

