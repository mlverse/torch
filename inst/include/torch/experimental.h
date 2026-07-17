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

enum class ScalarType {
  Float32,
  Float64,
  Float16,
  BFloat16,
  ComplexHalf,
  ComplexFloat,
  ComplexDouble,
  Float8E4M3FN,
  Float8E5M2,
  UInt8,
  Int8,
  Int16,
  Int32,
  Int64,
  Bool,
  QUInt8,
  QInt8,
  QInt32,
};

using Dtype = ScalarType;

inline constexpr ScalarType kFloat32 = ScalarType::Float32;
inline constexpr ScalarType kFloat64 = ScalarType::Float64;
inline constexpr ScalarType kFloat16 = ScalarType::Float16;
inline constexpr ScalarType kBFloat16 = ScalarType::BFloat16;
inline constexpr ScalarType kComplexHalf = ScalarType::ComplexHalf;
inline constexpr ScalarType kComplexFloat = ScalarType::ComplexFloat;
inline constexpr ScalarType kComplexDouble = ScalarType::ComplexDouble;
inline constexpr ScalarType kFloat8E4M3FN = ScalarType::Float8E4M3FN;
inline constexpr ScalarType kFloat8E5M2 = ScalarType::Float8E5M2;
inline constexpr ScalarType kUInt8 = ScalarType::UInt8;
inline constexpr ScalarType kInt8 = ScalarType::Int8;
inline constexpr ScalarType kInt16 = ScalarType::Int16;
inline constexpr ScalarType kInt32 = ScalarType::Int32;
inline constexpr ScalarType kInt64 = ScalarType::Int64;
inline constexpr ScalarType kBool = ScalarType::Bool;
inline constexpr ScalarType kQUInt8 = ScalarType::QUInt8;
inline constexpr ScalarType kQInt8 = ScalarType::QInt8;
inline constexpr ScalarType kQInt32 = ScalarType::QInt32;

// LibTorch's conventional aliases from torch/types.h.
inline constexpr ScalarType kByte = kUInt8;
inline constexpr ScalarType kChar = kInt8;
inline constexpr ScalarType kShort = kInt16;
inline constexpr ScalarType kInt = kInt32;
inline constexpr ScalarType kLong = kInt64;
inline constexpr ScalarType kHalf = kFloat16;
inline constexpr ScalarType kFloat = kFloat32;
inline constexpr ScalarType kDouble = kFloat64;
inline constexpr ScalarType kComplexFloat32 = kComplexHalf;
inline constexpr ScalarType kComplexFloat64 = kComplexFloat;
inline constexpr ScalarType kComplexFloat128 = kComplexDouble;

enum class Layout { Strided, Sparse, SparseCsr, SparseCsc, SparseBsr, SparseBsc };

inline constexpr Layout kStrided = Layout::Strided;
inline constexpr Layout kSparse = Layout::Sparse;
inline constexpr Layout kSparseCsr = Layout::SparseCsr;
inline constexpr Layout kSparseCsc = Layout::SparseCsc;
inline constexpr Layout kSparseBsr = Layout::SparseBsr;
inline constexpr Layout kSparseBsc = Layout::SparseBsc;

enum class DeviceType { CPU, CUDA };

inline constexpr DeviceType kCPU = DeviceType::CPU;
inline constexpr DeviceType kCUDA = DeviceType::CUDA;

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

inline ::torch::TensorOptions tensor_options() {
  return ::torch::TensorOptions(call<void*>("_lantern_TensorOptions"));
}

inline ::torch::IntArrayRef int_array_ref(
    const std::vector<std::int64_t>& values) {
  auto* data = values.empty() ? nullptr
                              : const_cast<std::int64_t*>(values.data());
  return ::torch::IntArrayRef(
      call<void*>("_lantern_vector_int64_t", data, values.size()));
}

inline ::torch::Scalar scalar(double value) {
  return ::torch::Scalar(call<void*>("_lantern_Scalar", &value, "double"));
}

inline ::torch::Scalar scalar(int value) {
  return ::torch::Scalar(call<void*>("_lantern_Scalar", &value, "int"));
}

inline ::torch::int64_t integer(std::int64_t value) {
  return ::torch::int64_t(call<void*>("_lantern_int64_t", value));
}

inline ::torch::double_t floating(double value) {
  return ::torch::double_t(call<void*>("_lantern_double", value));
}

inline ::torch::bool_t boolean(bool value) {
  return ::torch::bool_t(call<void*>("_lantern_bool", value));
}

inline XPtrTorchoptional_memory_format optional_memory_format() {
  return XPtrTorchoptional_memory_format(
      call<void*>("_lantern_optional_memory_format", nullptr));
}

inline ::torch::Dtype dtype(ScalarType type) {
  const char* symbol = nullptr;
  switch (type) {
    case ScalarType::Float32: symbol = "_lantern_Dtype_float32"; break;
    case ScalarType::Float64: symbol = "_lantern_Dtype_float64"; break;
    case ScalarType::Float16: symbol = "_lantern_Dtype_float16"; break;
    case ScalarType::BFloat16: symbol = "_lantern_Dtype_bfloat16"; break;
    case ScalarType::ComplexHalf: symbol = "_lantern_Dtype_chalf"; break;
    case ScalarType::ComplexFloat: symbol = "_lantern_Dtype_cfloat"; break;
    case ScalarType::ComplexDouble: symbol = "_lantern_Dtype_cdouble"; break;
    case ScalarType::Float8E4M3FN: symbol = "_lantern_Dtype_float8_e4m3fn"; break;
    case ScalarType::Float8E5M2: symbol = "_lantern_Dtype_float8_e5m2"; break;
    case ScalarType::UInt8: symbol = "_lantern_Dtype_uint8"; break;
    case ScalarType::Int8: symbol = "_lantern_Dtype_int8"; break;
    case ScalarType::Int16: symbol = "_lantern_Dtype_int16"; break;
    case ScalarType::Int32: symbol = "_lantern_Dtype_int32"; break;
    case ScalarType::Int64: symbol = "_lantern_Dtype_int64"; break;
    case ScalarType::Bool: symbol = "_lantern_Dtype_bool"; break;
    case ScalarType::QUInt8: symbol = "_lantern_Dtype_quint8"; break;
    case ScalarType::QInt8: symbol = "_lantern_Dtype_qint8"; break;
    case ScalarType::QInt32: symbol = "_lantern_Dtype_qint32"; break;
  }
  return ::torch::Dtype(call<void*>(symbol));
}

inline ::torch::Layout layout(Layout type) {
  const char* symbol = nullptr;
  switch (type) {
    case Layout::Strided: symbol = "_lantern_Layout_strided"; break;
    case Layout::Sparse: symbol = "_lantern_Layout_sparse"; break;
    case Layout::SparseCsr: symbol = "_lantern_Layout_sparse_csr"; break;
    case Layout::SparseCsc: symbol = "_lantern_Layout_sparse_csc"; break;
    case Layout::SparseBsr: symbol = "_lantern_Layout_sparse_bsr"; break;
    case Layout::SparseBsc: symbol = "_lantern_Layout_sparse_bsc"; break;
  }
  return ::torch::Layout(call<void*>(symbol));
}

inline ::torch::Device device(DeviceType type, std::int64_t index,
                              bool has_index) {
  const char* name = type == DeviceType::CUDA ? "cuda" : "cpu";
  return ::torch::Device(
      call<void*>("_lantern_Device", name, index, has_index));
}

}  // namespace detail

class TensorOptions {
 public:
  TensorOptions() : options_(detail::tensor_options()) {}

  TensorOptions dtype(ScalarType type) const {
    auto value = detail::dtype(type);
    return TensorOptions(detail::call<void*>(
        "_lantern_TensorOptions_dtype", options_.get(), value.get()));
  }

  TensorOptions layout(Layout type) const {
    auto value = detail::layout(type);
    return TensorOptions(detail::call<void*>(
        "_lantern_TensorOptions_layout", options_.get(), value.get()));
  }

  TensorOptions device(DeviceType type) const {
    auto value = detail::device(type, 0, false);
    return TensorOptions(detail::call<void*>(
        "_lantern_TensorOptions_device", options_.get(), value.get()));
  }

  TensorOptions device(DeviceType type, std::int64_t index) const {
    auto value = detail::device(type, index, true);
    return TensorOptions(detail::call<void*>(
        "_lantern_TensorOptions_device", options_.get(), value.get()));
  }

  TensorOptions requires_grad(bool value = true) const {
    return TensorOptions(detail::call<void*>(
        "_lantern_TensorOptions_requires_grad", options_.get(), value));
  }

  TensorOptions pinned_memory(bool value = true) const {
    return TensorOptions(detail::call<void*>(
        "_lantern_TensorOptions_pinned_memory", options_.get(), value));
  }

  const ::torch::TensorOptions& as_torch() const noexcept { return options_; }
  operator const ::torch::TensorOptions&() const noexcept { return options_; }

 private:
  explicit TensorOptions(void* value) : options_(value) {}
  ::torch::TensorOptions options_;
};

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

  Tensor add(double other) const {
    auto value = detail::scalar(other);
    auto alpha = detail::scalar(1);
    return from_raw(detail::call<void*>(
        "_lantern_Tensor_add_tensor_scalar_scalar", get(), value.get(),
        alpha.get()));
  }

  Tensor add_(const Tensor& other) {
    auto alpha = detail::scalar(1);
    return from_raw(detail::call<void*>(
        "_lantern_Tensor_add__tensor_tensor_scalar", get(), other.get(),
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

  Tensor mul(double other) const {
    auto value = detail::scalar(other);
    return from_raw(detail::call<void*>(
        "_lantern_Tensor_mul_tensor_scalar", get(), value.get()));
  }

  Tensor pow(double exponent) const {
    auto value = detail::scalar(exponent);
    return from_raw(detail::call<void*>(
        "_lantern_Tensor_pow_tensor_scalar", get(), value.get()));
  }

  Tensor fill_(double value) {
    auto scalar_value = detail::scalar(value);
    return from_raw(detail::call<void*>(
        "_lantern_Tensor_fill__tensor_scalar", get(), scalar_value.get()));
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
  Tensor operator+(double other) const { return add(other); }
  Tensor operator-(const Tensor& other) const { return sub(other); }
  Tensor operator*(const Tensor& other) const { return mul(other); }
  Tensor operator*(double other) const { return mul(other); }
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

inline Tensor cat(const std::vector<Tensor>& tensors, std::int64_t dim = 0) {
  ::torch::TensorList values(detail::call<void*>("_lantern_TensorList"));
  for (const auto& tensor : tensors) {
    detail::call_void("_lantern_TensorList_push_back", values.get(),
                      tensor.get());
  }
  auto dimension = detail::integer(dim);
  return Tensor(::torch::Tensor(detail::call<void*>(
      "_lantern_cat_constitensorlistref_intt", values.get(),
      dimension.get())));
}

inline Tensor cat(std::initializer_list<Tensor> tensors,
                  std::int64_t dim = 0) {
  return cat(std::vector<Tensor>(tensors), dim);
}

inline Tensor empty(const std::vector<std::int64_t>& size,
                    const ::torch::TensorOptions& options) {
  auto dimensions = detail::int_array_ref(size);
  auto memory_format = detail::optional_memory_format();
  return Tensor(::torch::Tensor(detail::call<void*>(
      "_lantern_empty_intarrayref_tensoroptions_memoryformat",
      dimensions.get(), options.get(), memory_format.get())));
}

inline Tensor empty(const std::vector<std::int64_t>& size) {
  return empty(size, detail::tensor_options());
}

inline Tensor empty(std::initializer_list<std::int64_t> size,
                    const ::torch::TensorOptions& options) {
  return empty(std::vector<std::int64_t>(size), options);
}

inline Tensor empty(std::initializer_list<std::int64_t> size) {
  return empty(std::vector<std::int64_t>(size));
}

#define TORCH_EXPERIMENTAL_SHAPED_CREATOR(name)                              \
  inline Tensor name(const std::vector<std::int64_t>& size,                  \
                     const ::torch::TensorOptions& options) {                \
    auto dimensions = detail::int_array_ref(size);                           \
    return Tensor(::torch::Tensor(detail::call<void*>(                       \
        "_lantern_" #name "_intarrayref_tensoroptions", dimensions.get(),  \
        options.get())));                                                    \
  }                                                                          \
  inline Tensor name(const std::vector<std::int64_t>& size) {                \
    return name(size, detail::tensor_options());                             \
  }                                                                          \
  inline Tensor name(std::initializer_list<std::int64_t> size,              \
                     const ::torch::TensorOptions& options) {                \
    return name(std::vector<std::int64_t>(size), options);                   \
  }                                                                          \
  inline Tensor name(std::initializer_list<std::int64_t> size) {            \
    return name(std::vector<std::int64_t>(size));                            \
  }

TORCH_EXPERIMENTAL_SHAPED_CREATOR(zeros)
TORCH_EXPERIMENTAL_SHAPED_CREATOR(ones)
TORCH_EXPERIMENTAL_SHAPED_CREATOR(rand)
TORCH_EXPERIMENTAL_SHAPED_CREATOR(randn)

#undef TORCH_EXPERIMENTAL_SHAPED_CREATOR

#define TORCH_EXPERIMENTAL_LIKE_CREATOR(name)                               \
  inline Tensor name##_like(const Tensor& input,                            \
                            const ::torch::TensorOptions& options) {         \
    auto memory_format = detail::optional_memory_format();                  \
    return Tensor(::torch::Tensor(detail::call<void*>(                      \
        "_lantern_" #name "_like_tensor_tensoroptions_memoryformat",      \
        input.get(), options.get(), memory_format.get())));                 \
  }                                                                         \
  inline Tensor name##_like(const Tensor& input) {                          \
    return name##_like(input, detail::tensor_options());                    \
  }

TORCH_EXPERIMENTAL_LIKE_CREATOR(empty)
TORCH_EXPERIMENTAL_LIKE_CREATOR(zeros)
TORCH_EXPERIMENTAL_LIKE_CREATOR(ones)
TORCH_EXPERIMENTAL_LIKE_CREATOR(rand)
TORCH_EXPERIMENTAL_LIKE_CREATOR(randn)

#undef TORCH_EXPERIMENTAL_LIKE_CREATOR

inline Tensor empty_strided(const std::vector<std::int64_t>& size,
                            const std::vector<std::int64_t>& stride,
                            const ::torch::TensorOptions& options) {
  auto dimensions = detail::int_array_ref(size);
  auto strides = detail::int_array_ref(stride);
  return Tensor(::torch::Tensor(detail::call<void*>(
      "_lantern_empty_strided_intarrayref_intarrayref_tensoroptions",
      dimensions.get(), strides.get(), options.get())));
}

inline Tensor empty_strided(const std::vector<std::int64_t>& size,
                            const std::vector<std::int64_t>& stride) {
  return empty_strided(size, stride, detail::tensor_options());
}

inline Tensor full(const std::vector<std::int64_t>& size, double fill_value,
                   const ::torch::TensorOptions& options) {
  auto dimensions = detail::int_array_ref(size);
  auto value = detail::scalar(fill_value);
  return Tensor(::torch::Tensor(detail::call<void*>(
      "_lantern_full_intarrayref_scalar_tensoroptions", dimensions.get(),
      value.get(), options.get())));
}

inline Tensor full(const std::vector<std::int64_t>& size, double fill_value) {
  return full(size, fill_value, detail::tensor_options());
}

inline Tensor full(std::initializer_list<std::int64_t> size, double fill_value,
                   const ::torch::TensorOptions& options) {
  return full(std::vector<std::int64_t>(size), fill_value, options);
}

inline Tensor full(std::initializer_list<std::int64_t> size,
                   double fill_value) {
  return full(std::vector<std::int64_t>(size), fill_value);
}

inline Tensor full_like(const Tensor& input, double fill_value,
                        const ::torch::TensorOptions& options) {
  auto value = detail::scalar(fill_value);
  auto memory_format = detail::optional_memory_format();
  return Tensor(::torch::Tensor(detail::call<void*>(
      "_lantern_full_like_tensor_scalar_tensoroptions_memoryformat",
      input.get(), value.get(), options.get(), memory_format.get())));
}

inline Tensor full_like(const Tensor& input, double fill_value) {
  return full_like(input, fill_value, detail::tensor_options());
}

inline Tensor randint(std::int64_t high,
                      const std::vector<std::int64_t>& size,
                      const ::torch::TensorOptions& options) {
  auto high_value = detail::integer(high);
  auto dimensions = detail::int_array_ref(size);
  return Tensor(::torch::Tensor(detail::call<void*>(
      "_lantern_randint_intt_intarrayref_tensoroptions", high_value.get(),
      dimensions.get(), options.get())));
}

inline Tensor randint(std::int64_t high,
                      const std::vector<std::int64_t>& size) {
  return randint(high, size, detail::tensor_options());
}

inline Tensor randint(std::int64_t low, std::int64_t high,
                      const std::vector<std::int64_t>& size,
                      const ::torch::TensorOptions& options) {
  auto low_value = detail::integer(low);
  auto high_value = detail::integer(high);
  auto dimensions = detail::int_array_ref(size);
  return Tensor(::torch::Tensor(detail::call<void*>(
      "_lantern_randint_intt_intt_intarrayref_tensoroptions", low_value.get(),
      high_value.get(), dimensions.get(), options.get())));
}

inline Tensor randint(std::int64_t low, std::int64_t high,
                      const std::vector<std::int64_t>& size) {
  return randint(low, high, size, detail::tensor_options());
}

inline Tensor randint_like(const Tensor& input, std::int64_t low,
                           std::int64_t high,
                           const ::torch::TensorOptions& options) {
  auto low_value = detail::integer(low);
  auto high_value = detail::integer(high);
  auto memory_format = detail::optional_memory_format();
  return Tensor(::torch::Tensor(detail::call<void*>(
      "_lantern_randint_like_tensor_intt_intt_tensoroptions_memoryformat",
      input.get(), low_value.get(), high_value.get(), options.get(),
      memory_format.get())));
}

inline Tensor randint_like(const Tensor& input, std::int64_t low,
                           std::int64_t high) {
  return randint_like(input, low, high, detail::tensor_options());
}

inline Tensor randperm(std::int64_t n,
                       const ::torch::TensorOptions& options) {
  auto count = detail::integer(n);
  return Tensor(::torch::Tensor(detail::call<void*>(
      "_lantern_randperm_intt_tensoroptions", count.get(), options.get())));
}

inline Tensor randperm(std::int64_t n) {
  return randperm(n, detail::tensor_options());
}

inline Tensor arange(double end, const ::torch::TensorOptions& options) {
  auto end_value = detail::scalar(end);
  return Tensor(::torch::Tensor(detail::call<void*>(
      "_lantern_arange_scalar_tensoroptions", end_value.get(),
      options.get())));
}

inline Tensor arange(double end) {
  return arange(end, detail::tensor_options());
}

inline Tensor arange(double start, double end, double step,
                     const ::torch::TensorOptions& options) {
  auto start_value = detail::scalar(start);
  auto end_value = detail::scalar(end);
  auto step_value = detail::scalar(step);
  return Tensor(::torch::Tensor(detail::call<void*>(
      "_lantern_arange_scalar_scalar_scalar_tensoroptions", start_value.get(),
      end_value.get(), step_value.get(), options.get())));
}

inline Tensor arange(double start, double end, double step = 1.0) {
  return arange(start, end, step, detail::tensor_options());
}

inline Tensor linspace(double start, double end, std::int64_t steps,
                       const ::torch::TensorOptions& options) {
  auto start_value = detail::scalar(start);
  auto end_value = detail::scalar(end);
  auto count = detail::integer(steps);
  return Tensor(::torch::Tensor(detail::call<void*>(
      "_lantern_linspace_scalar_scalar_intt_tensoroptions", start_value.get(),
      end_value.get(), count.get(), options.get())));
}

inline Tensor linspace(double start, double end, std::int64_t steps = 100) {
  return linspace(start, end, steps, detail::tensor_options());
}

inline Tensor logspace(double start, double end, std::int64_t steps,
                       double base, const ::torch::TensorOptions& options) {
  auto start_value = detail::scalar(start);
  auto end_value = detail::scalar(end);
  auto count = detail::integer(steps);
  auto base_value = detail::floating(base);
  return Tensor(::torch::Tensor(detail::call<void*>(
      "_lantern_logspace_scalar_scalar_intt_double_tensoroptions",
      start_value.get(), end_value.get(), count.get(), base_value.get(),
      options.get())));
}

inline Tensor logspace(double start, double end, std::int64_t steps = 100,
                       double base = 10.0) {
  return logspace(start, end, steps, base, detail::tensor_options());
}

inline Tensor eye(std::int64_t n, const ::torch::TensorOptions& options) {
  auto rows = detail::integer(n);
  return Tensor(::torch::Tensor(detail::call<void*>(
      "_lantern_eye_intt_tensoroptions", rows.get(), options.get())));
}

inline Tensor eye(std::int64_t n) {
  return eye(n, detail::tensor_options());
}

inline Tensor eye(std::int64_t n, std::int64_t m,
                  const ::torch::TensorOptions& options) {
  auto rows = detail::integer(n);
  auto columns = detail::integer(m);
  return Tensor(::torch::Tensor(detail::call<void*>(
      "_lantern_eye_intt_intt_tensoroptions", rows.get(), columns.get(),
      options.get())));
}

inline Tensor eye(std::int64_t n, std::int64_t m) {
  return eye(n, m, detail::tensor_options());
}

inline Tensor scalar_tensor(double value,
                            const ::torch::TensorOptions& options) {
  auto scalar = detail::scalar(value);
  return Tensor(::torch::Tensor(detail::call<void*>(
      "_lantern_scalar_tensor_scalar_tensoroptions", scalar.get(),
      options.get())));
}

inline Tensor scalar_tensor(double value) {
  return scalar_tensor(value, detail::tensor_options());
}

}  // namespace experimental

#include "experimental_namespace_functions.h"

// Opt-in public namespace spellings matching LibTorch's free-function API.
// Tensor itself remains under torch::experimental because torch.h already
// exposes the established Lantern handle as torch::Tensor.
using experimental::arange;
using experimental::cat;
using experimental::empty;
using experimental::empty_like;
using experimental::empty_strided;
using experimental::eye;
using experimental::full;
using experimental::full_like;
using experimental::linspace;
using experimental::logspace;
using experimental::ones;
using experimental::ones_like;
using experimental::rand;
using experimental::rand_like;
using experimental::randint;
using experimental::randint_like;
using experimental::randn;
using experimental::randn_like;
using experimental::randperm;
using experimental::scalar_tensor;
using experimental::zeros;
using experimental::zeros_like;

}  // namespace torch

#endif  // TORCH_EXPERIMENTAL_H
