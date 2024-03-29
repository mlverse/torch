% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/distributions-bernoulli.R
\name{distr_bernoulli}
\alias{distr_bernoulli}
\title{Creates a Bernoulli distribution parameterized by \code{probs}
or \code{logits} (but not both).
Samples are binary (0 or 1). They take the value \code{1} with probability \code{p}
and \code{0} with probability \code{1 - p}.}
\usage{
distr_bernoulli(probs = NULL, logits = NULL, validate_args = NULL)
}
\arguments{
\item{probs}{(numeric or torch_tensor): the probability of sampling \code{1}}

\item{logits}{(numeric or torch_tensor): the log-odds of sampling \code{1}}

\item{validate_args}{whether to validate arguments or not.}
}
\description{
Creates a Bernoulli distribution parameterized by \code{probs}
or \code{logits} (but not both).
Samples are binary (0 or 1). They take the value \code{1} with probability \code{p}
and \code{0} with probability \code{1 - p}.
}
\examples{
if (torch_is_installed()) {
m <- distr_bernoulli(0.3)
m$sample() # 30\% chance 1; 70\% chance 0
}
}
\seealso{
\link{Distribution} for details on the available methods.

Other distributions: 
\code{\link{distr_chi2}()},
\code{\link{distr_gamma}()},
\code{\link{distr_multivariate_normal}()},
\code{\link{distr_normal}()},
\code{\link{distr_poisson}()}
}
\concept{distributions}
