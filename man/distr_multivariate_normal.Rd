% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/distributions-multivariate_normal.R
\name{distr_multivariate_normal}
\alias{distr_multivariate_normal}
\title{Gaussian distribution}
\usage{
distr_multivariate_normal(
  loc,
  covariance_matrix = NULL,
  precision_matrix = NULL,
  scale_tril = NULL,
  validate_args = NULL
)
}
\arguments{
\item{loc}{(Tensor): mean of the distribution}

\item{covariance_matrix}{(Tensor): positive-definite covariance matrix}

\item{precision_matrix}{(Tensor): positive-definite precision matrix}

\item{scale_tril}{(Tensor): lower-triangular factor of covariance, with positive-valued diagonal}

\item{validate_args}{Bool wether to validate the arguments or not.}
}
\description{
Creates a multivariate normal (also called Gaussian) distribution
parameterized by a mean vector and a covariance matrix.
}
\details{
The multivariate normal distribution can be parameterized either
in terms of a positive definite covariance matrix \eqn{\mathbf{\Sigma}}
or a positive definite precision matrix \eqn{\mathbf{\Sigma}^{-1}}
or a lower-triangular matrix \eqn{\mathbf{L}} with positive-valued
diagonal entries, such that
\eqn{\mathbf{\Sigma} = \mathbf{L}\mathbf{L}^\top}. This triangular matrix
can be obtained via e.g. Cholesky decomposition of the covariance.
}
\note{
Only one of \code{covariance_matrix} or \code{precision_matrix} or
\code{scale_tril} can be specified.
Using \code{scale_tril} will be more efficient: all computations internally
are based on \code{scale_tril}. If \code{covariance_matrix} or
\code{precision_matrix} is passed instead, it is only used to compute
the corresponding lower triangular matrices using a Cholesky decomposition.
}
\examples{
if (torch_is_installed()) {
m <- distr_multivariate_normal(torch_zeros(2), torch_eye(2))
m$sample() # normally distributed with mean=`[0,0]` and covariance_matrix=`I`
}
}
\seealso{
\link{Distribution} for details on the available methods.

Other distributions: 
\code{\link{distr_bernoulli}()},
\code{\link{distr_chi2}()},
\code{\link{distr_gamma}()},
\code{\link{distr_normal}()},
\code{\link{distr_poisson}()}
}
\concept{distributions}
