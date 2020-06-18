#include "torch_types.h"
#include "utils.hpp"

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchPackedSequence> cpp_nn_utils_rnn_pack_padded_sequence (
    Rcpp::XPtr<XPtrTorchTensor> input, Rcpp::XPtr<XPtrTorchTensor> lengths,
    bool batch_first, bool enforce_sorted)
{
  XPtrTorchPackedSequence out = lantern_nn_utils_rnn_pack_padded_sequence(input->get(), lengths->get(), batch_first,
                                            enforce_sorted);
  return make_xptr<XPtrTorchPackedSequence>(out);
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchTensor> cpp_nn_utils_PackedSequence_data (Rcpp::XPtr<XPtrTorchPackedSequence> x)
{
  XPtrTorchTensor out = lantern_nn_utils_PackedSequence_data(x->get());
  return make_xptr<XPtrTorchTensor>(out);
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchTensor> cpp_nn_utils_PackedSequence_batch_sizes (Rcpp::XPtr<XPtrTorchPackedSequence> x)
{
  XPtrTorchTensor out = lantern_nn_utils_PackedSequence_batch_sizes(x->get());
  return make_xptr<XPtrTorchTensor>(out);
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchTensor> cpp_nn_utils_PackedSequence_sorted_indices (Rcpp::XPtr<XPtrTorchPackedSequence> x)
{
  XPtrTorchTensor out = lantern_nn_utils_PackedSequence_sorted_indices(x->get());
  return make_xptr<XPtrTorchTensor>(out);
}

// [[Rcpp::export]]
Rcpp::XPtr<XPtrTorchTensor> cpp_nn_utils_PackedSequence_unsorted_indices (Rcpp::XPtr<XPtrTorchPackedSequence> x)
{
  XPtrTorchTensor out = lantern_nn_utils_PackedSequence_unsorted_indices(x->get());
  return make_xptr<XPtrTorchTensor>(out);
}
