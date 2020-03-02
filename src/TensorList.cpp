#include <iostream>

#define LANTERN_BUILD

#include "lantern/lantern.h"

#include <torch/torch.h>

#include "utils.hpp"

void* lantern_TensorList () {
    auto out = new std::vector<torch::Tensor>;
    return (void*) out;
}

void lantern_TensorList_push_back(void* self, void* x) {
    auto slf = reinterpret_cast<std::vector<torch::Tensor>*>(self);
    torch::Tensor ten = reinterpret_cast<LanternObject<torch::Tensor>*>(x)->get();
    slf->push_back(ten);
}

void* lantern_TensorList_at (void* self, int64_t i) {
    auto slf = reinterpret_cast<std::vector<torch::Tensor>*>(self);
    torch::Tensor out = slf->at(i);
    return (void*) new LanternObject<torch::Tensor>(out);
}

int64_t lantern_TensorList_size (void* self) {
    auto slf = reinterpret_cast<std::vector<torch::Tensor>*>(self);
    return slf->size();
}