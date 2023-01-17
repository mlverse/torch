#include <iostream>

#define LANTERN_BUILD

#include <torch/torch.h>

#include "lantern/lantern.h"
#include "utils.hpp"

void *_lantern_nn_utils_rnn_pack_padded_sequence(void *input, void *lengths,
                                                 bool batch_first,
                                                 bool enforce_sorted) {
  LANTERN_FUNCTION_START
  auto out = torch::nn::utils::rnn::pack_padded_sequence(
      from_raw::Tensor(input), from_raw::Tensor(lengths), batch_first,
      enforce_sorted);

  return (void *)new LanternPtr<torch::nn::utils::rnn::PackedSequence>(out);
  LANTERN_FUNCTION_END
}

void *_lantern_nn_utils_rnn_pack_sequence(void *sequence, bool enforce_sorted) {
  LANTERN_FUNCTION_START
  auto out = torch::nn::utils::rnn::pack_sequence(
      from_raw::TensorList(sequence), enforce_sorted);

  return (void *)new LanternPtr<torch::nn::utils::rnn::PackedSequence>(out);
  LANTERN_FUNCTION_END
}

void *_lantern_nn_utils_rnn_pad_packed_sequence(void *sequence,
                                                bool batch_first,
                                                double padding_value,
                                                void *total_length) {
  LANTERN_FUNCTION_START
  auto out = torch::nn::utils::rnn::pad_packed_sequence(
      reinterpret_cast<LanternPtr<torch::nn::utils::rnn::PackedSequence> *>(
          sequence)
          ->get(),
      batch_first, padding_value, from_raw::optional::int64_t(total_length));

  std::vector<torch::Tensor> x;
  x.push_back(std::get<0>(out));
  x.push_back(std::get<1>(out));

  return make_raw::TensorList(x);
  LANTERN_FUNCTION_END
}

void *_lantern_nn_utils_rnn_pad_sequence(void *sequence, bool batch_first,
                                         double padding_value) {
  LANTERN_FUNCTION_START
  auto out = torch::nn::utils::rnn::pad_sequence(from_raw::TensorList(sequence),
                                                 batch_first, padding_value);

  return make_raw::Tensor(out);
  LANTERN_FUNCTION_END
}

void *_lantern_nn_utils_rnn_PackedSequence_new(void *data, void *batch_sizes,
                                               void *sorted_indices,
                                               void *unsorted_indices) {
  LANTERN_FUNCTION_START
  auto out = torch::nn::utils::rnn::PackedSequence(
      from_raw::Tensor(data), from_raw::Tensor(batch_sizes),
      from_raw::Tensor(sorted_indices), from_raw::Tensor(unsorted_indices));
  return (void *)new LanternPtr<torch::nn::utils::rnn::PackedSequence>(out);
  LANTERN_FUNCTION_END
}

void *_lantern_nn_utils_PackedSequence_data(void *input) {
  LANTERN_FUNCTION_START
  auto x =
      reinterpret_cast<LanternPtr<torch::nn::utils::rnn::PackedSequence> *>(
          input)
          ->get();
  return make_raw::Tensor(x.data());
  LANTERN_FUNCTION_END
}

void *_lantern_nn_utils_PackedSequence_batch_sizes(void *input) {
  LANTERN_FUNCTION_START
  auto x =
      reinterpret_cast<LanternPtr<torch::nn::utils::rnn::PackedSequence> *>(
          input)
          ->get();
  return make_raw::Tensor(x.batch_sizes());
  LANTERN_FUNCTION_END
}

void *_lantern_nn_utils_PackedSequence_sorted_indices(void *input) {
  LANTERN_FUNCTION_START
  auto x =
      reinterpret_cast<LanternPtr<torch::nn::utils::rnn::PackedSequence> *>(
          input)
          ->get();
  return make_raw::Tensor(x.sorted_indices());
  LANTERN_FUNCTION_END
}

void *_lantern_nn_utils_PackedSequence_unsorted_indices(void *input) {
  LANTERN_FUNCTION_START
  auto x =
      reinterpret_cast<LanternPtr<torch::nn::utils::rnn::PackedSequence> *>(
          input)
          ->get();
  return make_raw::Tensor(x.unsorted_indices());
  LANTERN_FUNCTION_END
}