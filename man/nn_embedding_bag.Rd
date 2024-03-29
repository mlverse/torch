% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/nn-sparse.R
\name{nn_embedding_bag}
\alias{nn_embedding_bag}
\title{Embedding bag module}
\usage{
nn_embedding_bag(
  num_embeddings,
  embedding_dim,
  max_norm = NULL,
  norm_type = 2,
  scale_grad_by_freq = FALSE,
  mode = "mean",
  sparse = FALSE,
  include_last_offset = FALSE,
  padding_idx = NULL,
  .weight = NULL
)
}
\arguments{
\item{num_embeddings}{(int): size of the dictionary of embeddings}

\item{embedding_dim}{(int): the size of each embedding vector}

\item{max_norm}{(float, optional): If given, each embedding vector with norm larger than \code{max_norm}
is renormalized to have norm \code{max_norm}.}

\item{norm_type}{(float, optional): The p of the p-norm to compute for the \code{max_norm} option. Default \code{2}}

\item{scale_grad_by_freq}{(boolean, optional): If given, this will scale gradients by the inverse of frequency of
the words in the mini-batch. Default \code{False}.}

\item{mode}{(string, optional): \code{"sum"}, \code{"mean"} or \code{"max"}. Specifies the way to reduce the bag.
\code{"sum"} computes the weighted sum, taking \code{per_sample_weights}  into consideration. \code{"mean"} computes
the average of the values in the bag, \code{"max"} computes the max value over each bag.}

\item{sparse}{(bool, optional): If \code{True}, gradient w.r.t. \code{weight} matrix will be a sparse tensor.
See Notes for more details regarding sparse gradients.}

\item{include_last_offset}{(bool, optional): if \code{True}, \code{offsets} has one additional element, where the last element
is equivalent to the size of \code{indices}. This matches the CSR format.}

\item{padding_idx}{(int, optional):  If given, pads the output with the embedding vector at \code{padding_idx}
(initialized to zeros) whenever it encounters the index.}

\item{.weight}{(Tensor, optional) embeddings weights (in case you want to set it manually)}
}
\description{
Computes sums, means or maxes of \code{bags} of embeddings, without instantiating the
intermediate embeddings.
}
\section{Attributes}{

\itemize{
\item weight (Tensor): the learnable weights of the module of shape (num_embeddings, embedding_dim)
initialized from \eqn{\mathcal{N}(0, 1)}
}
}

\examples{
if (torch_is_installed()) {
# an EmbeddingBag module containing 10 tensors of size 3
embedding_sum <- nn_embedding_bag(10, 3, mode = 'sum')
# a batch of 2 samples of 4 indices each
input <- torch_tensor(c(1, 2, 4, 5, 4, 3, 2, 9), dtype = torch_long())
offsets <- torch_tensor(c(0, 4), dtype = torch_long())
embedding_sum(input, offsets)
# example with padding_idx
embedding_sum <- nn_embedding_bag(10, 3, mode = 'sum', padding_idx = 1)
input <- torch_tensor(c(2, 2, 2, 2, 4, 3, 2, 9), dtype = torch_long())
offsets <- torch_tensor(c(0, 4), dtype = torch_long())
embedding_sum(input, offsets)
# An EmbeddingBag can be loaded from an Embedding like so
embedding <- nn_embedding(10, 3, padding_idx = 2)
embedding_sum <- nn_embedding_bag$from_pretrained(embedding$weight,
                                                 padding_idx = embedding$padding_idx,
                                                 mode='sum')
}
}
