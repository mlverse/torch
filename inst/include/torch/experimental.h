#ifndef TORCH_EXPERIMENTAL_H
#define TORCH_EXPERIMENTAL_H

#include <torch.h>

#include <cstdint>
#include <initializer_list>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace torch {
namespace experimental {

namespace detail {

template <typename Return, typename... Args>
Return call(const char* name, Args... args) {
  using Function = Return (*)(Args...);
  auto function = reinterpret_cast<Function>(lantern_get_symbol(name));
  auto result = function(args...);
  lantern_call_host_handler();
  return result;
}

template <typename... Args>
void call_void(const char* name, Args... args) {
  using Function = void (*)(Args...);
  auto function = reinterpret_cast<Function>(lantern_get_symbol(name));
  function(args...);
  lantern_call_host_handler();
}

}  // namespace detail

// An experimental, LibTorch-inspired facade over the Lantern C API.
//
// Tensor deliberately owns the existing torch::Tensor handle instead of the
// underlying LibTorch object. This keeps the Lantern ABI boundary intact and
// makes conversion to and from the C++ types already exposed by torch zero-copy.
class Tensor {
 public:
  Tensor() = default;
  explicit Tensor(::torch::Tensor tensor) : tensor_(std::move(tensor)) {}
  explicit Tensor(SEXP tensor) : tensor_(from_sexp_tensor(tensor)) {}

  operator SEXP() const { return operator_sexp_tensor(&tensor_); }

  void* get() const noexcept { return tensor_.get(); }
  bool has_value() const noexcept { return tensor_.get() != nullptr; }
  explicit operator bool() const noexcept { return has_value(); }

  const ::torch::Tensor& as_torch() const noexcept { return tensor_; }
  ::torch::Tensor& as_torch() noexcept { return tensor_; }

  std::int64_t dim() const {
    return detail::call<std::int64_t>("_lantern_Tensor_ndimension", get());
  }
  std::int64_t ndimension() const { return dim(); }
  std::int64_t numel() const {
    return detail::call<std::int64_t>("_lantern_Tensor_numel", get());
  }
  std::int64_t element_size() const {
    return detail::call<std::int64_t>("_lantern_Tensor_element_size", get());
  }
  std::int64_t size(std::int64_t dimension) const {
    return detail::call<std::int64_t>("_lantern_Tensor_size", get(), dimension);
  }

  std::vector<std::int64_t> sizes() const {
    std::vector<std::int64_t> result;
    const auto dimensions = dim();
    result.reserve(static_cast<std::size_t>(dimensions));
    for (std::int64_t i = 0; i < dimensions; ++i) {
      result.push_back(size(i));
    }
    return result;
  }

  bool requires_grad() const {
    return detail::call<bool>("_lantern_Tensor_requires_grad", get());
  }
  bool is_contiguous() const {
    return detail::call<bool>("_lantern_Tensor_is_contiguous", get());
  }
  bool is_quantized() const {
    return detail::call<bool>("_lantern_Tensor_is_quantized", get());
  }
  bool is_sparse() const {
    return detail::call<bool>("_lantern_Tensor_is_sparse", get());
  }
  bool is_sparse_csr() const {
    return detail::call<bool>("_lantern_Tensor_is_sparse_csr", get());
  }
  bool is_undefined() const {
    return detail::call<bool>("_lantern_Tensor_is_undefined", get());
  }
  bool defined() const { return has_value() && !is_undefined(); }
  bool has_storage() const {
    return detail::call<bool>("_lantern_Tensor_has_storage", get());
  }
  bool has_names() const {
    return detail::call<bool>("_lantern_Tensor_has_names", get());
  }
  bool has_any_zeros() const {
    return detail::call<bool>("_lantern_Tensor_has_any_zeros", get());
  }

  template <typename T>
  T* data_ptr() const {
    if constexpr (std::is_same<T, double>::value) {
      return detail::call<double*>("_lantern_Tensor_data_ptr_double", get());
    } else if constexpr (std::is_same<T, std::uint8_t>::value) {
      return detail::call<std::uint8_t*>(
          "_lantern_Tensor_data_ptr_uint8_t", get());
    } else if constexpr (std::is_same<T, std::int16_t>::value) {
      return detail::call<std::int16_t*>(
          "_lantern_Tensor_data_ptr_int16_t", get());
    } else if constexpr (std::is_same<T, std::int32_t>::value) {
      return detail::call<std::int32_t*>(
          "_lantern_Tensor_data_ptr_int32_t", get());
    } else if constexpr (std::is_same<T, std::int64_t>::value) {
      return detail::call<std::int64_t*>(
          "_lantern_Tensor_data_ptr_int64_t", get());
    } else if constexpr (std::is_same<T, bool>::value) {
      return detail::call<bool*>("_lantern_Tensor_data_ptr_bool", get());
    } else {
      static_assert(!std::is_same<T, T>::value,
                    "data_ptr<T>() is not supported for this dtype by Lantern");
    }
  }

  ::torch::Dtype dtype() const {
    return ::torch::Dtype(
        detail::call<void*>("_lantern_Tensor_dtype", get()));
  }

  ::torch::Device device() const {
    return ::torch::Device(
        detail::call<void*>("_lantern_Tensor_device", get()));
  }

  ::torch::Storage storage() const {
    return ::torch::Storage(
        detail::call<void*>("_lantern_Tensor_storage", get()));
  }

  ::torch::DimnameList names() const {
    return ::torch::DimnameList(
        detail::call<void*>("_lantern_Tensor_names", get()));
  }

  Tensor grad() const {
    return from_raw(detail::call<void*>("_lantern_Tensor_grad", get()));
  }
  Tensor clone() const {
    return from_raw(detail::call<void*>("_lantern_Tensor_clone", get()));
  }
  Tensor contiguous() const {
    return from_raw(detail::call<void*>("_lantern_Tensor_contiguous", get()));
  }
  Tensor detach() const {
    return from_raw(
        detail::call<void*>("_lantern_Tensor_detach_tensor", get()));
  }
  Tensor relu() const {
    return from_raw(detail::call<void*>("_lantern_Tensor_relu_tensor", get()));
  }
  Tensor sigmoid() const {
    return from_raw(
        detail::call<void*>("_lantern_Tensor_sigmoid_tensor", get()));
  }
  Tensor neg() const {
    return from_raw(detail::call<void*>("_lantern_Tensor_neg_tensor", get()));
  }
  Tensor t() const {
    return from_raw(detail::call<void*>("_lantern_Tensor_t_tensor", get()));
  }

  Tensor matmul(const Tensor& other) const {
    return from_raw(detail::call<void*>(
        "_lantern_Tensor_matmul_tensor_tensor", get(), other.get()));
  }

  Tensor add(const Tensor& other) const {
    auto alpha = scalar(1);
    return from_raw(detail::call<void*>(
        "_lantern_Tensor_add_tensor_tensor_scalar", get(), other.get(),
        alpha.get()));
  }

  Tensor sub(const Tensor& other) const {
    auto alpha = scalar(1);
    return from_raw(detail::call<void*>(
        "_lantern_Tensor_sub_tensor_tensor_scalar", get(), other.get(),
        alpha.get()));
  }

  Tensor mul(const Tensor& other) const {
    return from_raw(detail::call<void*>(
        "_lantern_Tensor_mul_tensor_tensor", get(), other.get()));
  }

  Tensor div(const Tensor& other) const {
    return from_raw(detail::call<void*>(
        "_lantern_Tensor_div_tensor_tensor", get(), other.get()));
  }

  Tensor reshape(const std::vector<std::int64_t>& shape) const {
    auto dimensions = int_array_ref(shape);
    return from_raw(detail::call<void*>(
        "_lantern_Tensor_reshape_tensor_intarrayref", get(), dimensions.get()));
  }

  Tensor reshape(std::initializer_list<std::int64_t> shape) const {
    return reshape(std::vector<std::int64_t>(shape));
  }

  Tensor view(const std::vector<std::int64_t>& shape) const {
    auto dimensions = int_array_ref(shape);
    return from_raw(detail::call<void*>(
        "_lantern_Tensor_view_tensor_intarrayref", get(), dimensions.get()));
  }

  Tensor view(std::initializer_list<std::int64_t> shape) const {
    return view(std::vector<std::int64_t>(shape));
  }

  Tensor permute(const std::vector<std::int64_t>& dimensions) const {
    auto dims = int_array_ref(dimensions);
    return from_raw(detail::call<void*>(
        "_lantern_Tensor_permute", get(), dims.get()));
  }

  Tensor permute(std::initializer_list<std::int64_t> dimensions) const {
    return permute(std::vector<std::int64_t>(dimensions));
  }

  Tensor transpose(std::int64_t dim0, std::int64_t dim1) const {
    auto first = integer(dim0);
    auto second = integer(dim1);
    return from_raw(detail::call<void*>(
        "_lantern_Tensor_transpose_tensor_intt_intt", get(), first.get(),
        second.get()));
  }

  Tensor to(const ::torch::TensorOptions& options) const {
    return from_raw(detail::call<void*>(
        "_lantern_Tensor_to", get(), options.get()));
  }

  Tensor& requires_grad_(bool requires_grad = true) {
    // Lantern returns another owning wrapper around the mutated tensor.
    // Letting it die here releases only that wrapper, not the tensor value.
    auto value = boolean(requires_grad);
    ::torch::Tensor result(detail::call<void*>(
        "_lantern_Tensor_requires_grad__tensor_bool", get(), value.get()));
    return *this;
  }

  Tensor& relu_() {
    ::torch::Tensor result(
        detail::call<void*>("_lantern_Tensor_relu__tensor", get()));
    return *this;
  }

  Tensor& zero_() {
    ::torch::Tensor result(
        detail::call<void*>("_lantern_Tensor_zero__tensor", get()));
    return *this;
  }

  Tensor& set_grad_(const Tensor& gradient) {
    detail::call_void("_lantern_Tensor_set_grad_", get(), gradient.get());
    return *this;
  }

  Tensor index(const ::torch::indexing::TensorIndex& indices) const {
    return from_raw(detail::call<void*>(
        "_lantern_Tensor_index", get(), indices.get()));
  }

  Tensor& index_put_(const ::torch::indexing::TensorIndex& indices,
                     const Tensor& value) {
    detail::call_void(
        "_lantern_Tensor_index_put_tensor_", get(), indices.get(), value.get());
    return *this;
  }

  Tensor& index_put_(const ::torch::indexing::TensorIndex& indices,
                     const ::torch::Scalar& value) {
    detail::call_void(
        "_lantern_Tensor_index_put_scalar_", get(), indices.get(), value.get());
    return *this;
  }

  static Tensor undefined() {
    return from_raw(detail::call<void*>("_lantern_Tensor_undefined"));
  }

  std::string to_string() const {
    const char* value = detail::call<const char*>(
        "_lantern_Tensor_StreamInsertion", get());
    std::string result(value);
    detail::call_void("_lantern_const_char_delete", value);
    return result;
  }

  XPtrTorch grad_fn() const {
    return XPtrTorch(detail::call<void*>("_lantern_Tensor_grad_fn", get()));
  }

  unsigned int register_hook(const XPtrTorchFunctionPtr& hook) {
    return detail::call<unsigned int>(
        "_lantern_Tensor_register_hook", get(), hook.get());
  }

  void remove_hook(unsigned int position) {
    detail::call_void("_lantern_Tensor_remove_hook", get(), position);
  }

  Tensor _backward(const ::torch::TensorList& inputs,
                   const Tensor& gradient,
                   const ::torch::bool_t& retain_graph,
                   const ::torch::bool_t& create_graph) const {
    return from_raw(detail::call<void*>(
        "_lantern_Tensor__backward_tensor_tensorlist_tensor_bool_bool", get(),
        inputs.get(), gradient.get(), retain_graph.get(), create_graph.get()));
  }

  Tensor std(const ::torch::IntArrayRef& dimensions,
             const ::torch::Scalar& correction,
             const ::torch::bool_t& keepdim) const {
    return from_raw(detail::call<void*>(
        "_lantern_Tensor_std_tensor_intarrayref_scalar_bool", get(),
        dimensions.get(), correction.get(), keepdim.get()));
  }

  Tensor std(const ::torch::DimnameList& dimensions,
             const ::torch::Scalar& correction,
             const ::torch::bool_t& keepdim) const {
    return from_raw(detail::call<void*>(
        "_lantern_Tensor_std_tensor_dimnamelist_scalar_bool", get(),
        dimensions.get(), correction.get(), keepdim.get()));
  }

  Tensor var(const ::torch::IntArrayRef& dimensions,
             const ::torch::Scalar& correction,
             const ::torch::bool_t& keepdim) const {
    return from_raw(detail::call<void*>(
        "_lantern_Tensor_var_tensor_intarrayref_scalar_bool", get(),
        dimensions.get(), correction.get(), keepdim.get()));
  }

  Tensor var(const ::torch::DimnameList& dimensions,
             const ::torch::Scalar& correction,
             const ::torch::bool_t& keepdim) const {
    return from_raw(detail::call<void*>(
        "_lantern_Tensor_var_tensor_dimnamelist_scalar_bool", get(),
        dimensions.get(), correction.get(), keepdim.get()));
  }

  Tensor stft(const ::torch::int64_t& n_fft,
              const ::torch::int64_t& hop_length,
              const ::torch::int64_t& win_length,
              const Tensor& window,
              const ::torch::bool_t& normalized,
              const ::torch::bool_t& onesided,
              const ::torch::bool_t& return_complex,
              const ::torch::bool_t& align_to_window) const {
    return from_raw(detail::call<void*>(
        "_lantern_Tensor_stft_tensor_intt_intt_intt_tensor_bool_bool_bool_bool",
        get(), n_fft.get(), hop_length.get(), win_length.get(), window.get(),
        normalized.get(), onesided.get(), return_complex.get(),
        align_to_window.get()));
  }

  Tensor operator-() const { return neg(); }
  Tensor operator+(const Tensor& other) const { return add(other); }
  Tensor operator-(const Tensor& other) const { return sub(other); }
  Tensor operator*(const Tensor& other) const { return mul(other); }
  Tensor operator/(const Tensor& other) const { return div(other); }

#include "experimental_tensor_methods.h"

 private:
  static Tensor from_raw(void* value) {
    return Tensor(::torch::Tensor(value));
  }

  static ::torch::IntArrayRef int_array_ref(
      const std::vector<std::int64_t>& values) {
    // Lantern copies the vector into its own owning argument wrapper.
    auto* data = values.empty()
                     ? nullptr
                     : const_cast<std::int64_t*>(values.data());
    return ::torch::IntArrayRef(detail::call<void*>(
        "_lantern_vector_int64_t", data, values.size()));
  }

  static ::torch::Scalar scalar(int value) {
    return ::torch::Scalar(
        detail::call<void*>("_lantern_Scalar", &value, "int"));
  }

  static ::torch::int64_t integer(std::int64_t value) {
    return ::torch::int64_t(
        detail::call<void*>("_lantern_int64_t", value));
  }

  static ::torch::bool_t boolean(bool value) {
    return ::torch::bool_t(detail::call<void*>("_lantern_bool", value));
  }

  static ::torch::double_t floating(double value) {
    return ::torch::double_t(detail::call<void*>("_lantern_double", value));
  }

  ::torch::Tensor tensor_;
};

inline Tensor from_torch(::torch::Tensor tensor) {
  return Tensor(std::move(tensor));
}

inline const ::torch::Tensor& as_torch(const Tensor& tensor) noexcept {
  return tensor.as_torch();
}

inline ::torch::Tensor& as_torch(Tensor& tensor) noexcept {
  return tensor.as_torch();
}

inline Tensor matmul(const Tensor& left, const Tensor& right) {
  return left.matmul(right);
}

}  // namespace experimental
}  // namespace torch

#endif  // TORCH_EXPERIMENTAL_H
