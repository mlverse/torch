#include <iostream>

#define LANTERN_BUILD

#include <torch/torch.h>

#include "base64.hpp"
#include "lantern/lantern.h"
#include "utils.hpp"

std::string size_t_to_string(std::size_t i) {
  std::stringstream ss;
  ss << std::setw(20) << std::setfill('0') << i;
  return ss.str();
}

const char* _lantern_tensor_save(void* self) {
  LANTERN_FUNCTION_START
  auto t = from_raw::Tensor(self);

  std::ostringstream oss;
  torch::save(t, oss);

  auto str = std::string(macaron::Base64::Encode(oss.str()));

  char* cstr = new char[str.length() + 1];
  strcpy(cstr, str.c_str());
  return cstr;
  LANTERN_FUNCTION_END
}

std::size_t _lantern_tensor_serialized_size(const char* s) {
  LANTERN_FUNCTION_START
  std::istringstream iss_size(std::string(s, 20));
  std::size_t size;
  iss_size >> size;
  return size;
  LANTERN_FUNCTION_END_RET(0)
}

void* _lantern_tensor_load(const char* s, void* device) {
  LANTERN_FUNCTION_START
  std::string str;
  macaron::Base64::Decode(std::string(s), str);
  std::istringstream stream(str);

  torch::Tensor t;
  c10::optional<torch::Device> device_ = from_raw::optional::Device(device);
  torch::load(t, stream, device_);
  return make_raw::Tensor(t);
  LANTERN_FUNCTION_END
}

void* _lantern_test_tensor() {
  LANTERN_FUNCTION_START
  return make_raw::Tensor(torch::ones({5, 5}));
  LANTERN_FUNCTION_END
}

void _lantern_test_print(void* x) {
  LANTERN_FUNCTION_START
  auto t = from_raw::Tensor(x);
  std::cout << t << std::endl;
  LANTERN_FUNCTION_END_VOID
}

void* _lantern_load_state_dict(const char* path) {
  LANTERN_FUNCTION_START
  std::ifstream file(path, std::ios::binary);
  std::vector<char> data((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());
  torch::IValue ivalue = torch::pickle_load(data);
  return make_raw::IValue(ivalue);
  LANTERN_FUNCTION_END
}

void* _lantern_get_state_dict_keys(void* ivalue) {
  LANTERN_FUNCTION_START
  auto iv = from_raw::IValue(ivalue);
  auto d = iv.toGenericDict();
  auto keys = new std::vector<std::string>;
  for (auto i = d.begin(); i != d.end(); ++i) {
    std::string key = i->key().toString()->string();
    keys->push_back(key);
  }
  return (void*)keys;
  LANTERN_FUNCTION_END
}

void* _lantern_get_state_dict_values(void* ivalue) {
  LANTERN_FUNCTION_START
  auto iv = from_raw::IValue(ivalue);
  auto d = iv.toGenericDict();
  std::vector<torch::Tensor> values;
  for (auto i = d.begin(); i != d.end(); ++i) {
    torch::Tensor value = i->value().toTensor();
    values.push_back(value);
  }
  return make_raw::TensorList(values);
  LANTERN_FUNCTION_END
}
