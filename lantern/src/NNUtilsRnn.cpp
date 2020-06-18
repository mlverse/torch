#include <iostream>

#define LANTERN_BUILD

#include "lantern/lantern.h"

#include <torch/torch.h>

#include "utils.hpp"

void *lantern_nn_utils_rnn_pack_padded_sequence(void *input, void *lengths, bool batch_first, bool enforce_sorted)
{
    auto out = torch::nn::utils::rnn::pack_padded_sequence(
        reinterpret_cast<LanternObject<torch::Tensor> *>(input)->get(),
        reinterpret_cast<LanternObject<torch::Tensor> *>(lengths)->get(),
        batch_first,
        enforce_sorted);

    return (void *)new LanternPtr<torch::nn::utils::rnn::PackedSequence>(out);
}

void *lantern_nn_utils_PackedSequence_data(void *input)
{
    auto x = reinterpret_cast<LanternPtr<torch::nn::utils::rnn::PackedSequence> *>(input)->get();
    return (void *)new LanternObject<torch::Tensor>(x.data());
}

void *lantern_nn_utils_PackedSequence_batch_sizes(void *input)
{
    auto x = reinterpret_cast<LanternPtr<torch::nn::utils::rnn::PackedSequence> *>(input)->get();
    return (void *)new LanternObject<torch::Tensor>(x.batch_sizes());
}

void *lantern_nn_utils_PackedSequence_sorted_indices(void *input)
{
    auto x = reinterpret_cast<LanternPtr<torch::nn::utils::rnn::PackedSequence> *>(input)->get();
    return (void *)new LanternObject<torch::Tensor>(x.sorted_indices());
}

void *lantern_nn_utils_PackedSequence_unsorted_indices(void *input)
{
    auto x = reinterpret_cast<LanternPtr<torch::nn::utils::rnn::PackedSequence> *>(input)->get();
    return (void *)new LanternObject<torch::Tensor>(x.unsorted_indices());
}