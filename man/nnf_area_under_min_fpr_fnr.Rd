% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/nnf-loss.R
\name{nnf_area_under_min_fpr_fnr}
\alias{nnf_area_under_min_fpr_fnr}
\title{Area under the \eqn{Min(FPR, FNR)} (AUM)}
\usage{
nnf_area_under_min_fpr_fnr(input, target)
}
\arguments{
\item{input}{Tensor of arbitrary shape}

\item{target}{Tensor of the same shape as input. Should be the factor
level of the binary outcome, i.e. with values \code{1L} and \code{2L}.}
}
\description{
Function that measures Area under the \eqn{Min(FPR, FNR)} (AUM)  between each
element in the \eqn{input} and \eqn{target}.
}
\details{
This is used for measuring the error of a binary reconstruction within highly unbalanced dataset,
where the goal is optimizing the ROC curve.
}
