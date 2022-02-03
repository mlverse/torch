#include <torch/torch.h>

// https://pt.stackoverflow.com/a/438284/6036
//
template <class... T>
std::vector<void*> to_vector(std::tuple<T...> x) {
  std::vector<void*> out;
  out.reserve(sizeof...(T));
  std::apply(
      [&out](auto&&... args) {
        ((out.push_back(new std::remove_reference_t<decltype(args)>(
             std::forward<decltype(args)>(args)))),
         ...);
      },
      x);
  return out;
}

// a template vector that can be easily converted to array<>
// with a few default casting for a few different sizes.
template <typename T>
class Vector {
 public:
  Vector(std::vector<T> x) : x_(x) {}
  Vector() : x_() {}
  std::vector<T> x_;
  operator std::array<T, 2>() const { return std::array<T, 2>{x_[0], x_[1]}; }
  operator std::array<T, 3>() const {
    return std::array<T, 3>{x_[0], x_[1], x_[2]};
  }
  operator std::array<T, 4>() const {
    return std::array<T, 4>{x_[0], x_[1], x_[2], x_[3]};
  }
  operator std::vector<T>() const { return x_; }
  void push_back(T x) { x_.push_back(x); }
  size_t size() const { return x_.size(); }
  T at(size_t i) const { return x_.at(i); }
};

template <class T>
void* make_ptr(const T& x) {
  return (void*)std::make_unique<T>(x).release();
}

template <class T>
void* make_ptr() {
  return (void*)std::make_unique<T>().release();
}

namespace make_raw {
void* Tensor(const torch::Tensor& x);
void* TensorList(const torch::TensorList& x);
void* ScalarType(const torch::ScalarType& x);
void* Scalar(const torch::Scalar& x);
void* TensorOptions(const torch::TensorOptions& x);
void* Device(torch::Device& x);
void* Dtype(const torch::Dtype& x);
void* Dimname(torch::Dimname& x);
void* DimnameList(const torch::DimnameList& x);
void* Generator(const torch::Generator& x);
void* MemoryFormat(const torch::MemoryFormat& x);
void* IntArrayRef(const torch::IntArrayRef& x);
void* IntArrayRef(const torch::IntArrayRef& x);
void* TensorDict(const c10::Dict<std::string, torch::Tensor>& x);
void* CompilationUnit(torch::jit::CompilationUnit& x);
void* QScheme(const torch::QScheme& x);
void* variable_list(const torch::autograd::variable_list& x);
void* Layout(const torch::Layout& x);
void* Storage(const torch::Storage& x);
void* string(const std::string& x);
void* string_view(const c10::string_view& x);
void* int64_t(const std::int64_t& x);
void* bool_t(const bool& x);
void* double_t(const double& x);
void* Stream(const at::Stream& x);
void* IValue(const torch::IValue& x);

namespace vector {
void* string(const std::vector<std::string>& x);
void* string();
void* int64_t(const std::vector<std::int64_t>& x);
void* int64_t();
void* double_t(const std::vector<double>& x);
void* double_t();
void* bool_t(const std::vector<bool>& x);
void* bool_t();
void* Scalar(const std::vector<torch::Scalar>& x);
}  // namespace vector

template <class... T>
void* tuple(std::tuple<T...> x) {
  return make_ptr<std::vector<void*>>(to_vector(x));
}

namespace optional {
void* bool_t(const c10::optional<bool>& x);
void* string(const c10::optional<std::string>& x);
void* string_view(const c10::optional<c10::string_view>& x);
void* TensorList(const c10::List<c10::optional<torch::Tensor>>& x);
void* IntArrayRef(const c10::optional<torch::IntArrayRef>& x);
void* DoubleArrayRef(const c10::optional<torch::ArrayRef<double>>& x);
void* Scalar(const c10::optional<torch::Scalar>& x);
void* double_t(const c10::optional<double>& x);
void* int64_t(const c10::optional<std::int64_t>& x);
void* DimnameList(const c10::optional<torch::DimnameList>& x);
void* Generator(const c10::optional<torch::Generator>& x);
void* Tensor(const c10::optional<torch::Tensor>& x);
void* ScalarType(const c10::optional<torch::ScalarType>& x);
void* MemoryFormat(const c10::optional<torch::MemoryFormat>& x);
void* Device(const c10::optional<torch::Device>& x);
}  // namespace optional

}  // namespace make_raw

#define LANTERN_FROM_RAW_DECL(name, type) type& name(void* x);

namespace alias {
using TensorDict = c10::Dict<std::string, torch::Tensor>;
}

namespace from_raw {
LANTERN_FROM_RAW_DECL(Tensor, torch::Tensor)
// TensorLists are passed as std::vector<torch::Tensor> because they don't own
// the underlying memory. Passing them as vectors is also fine as they are
// trivially constructed from them.
LANTERN_FROM_RAW_DECL(TensorList, torch::TensorList)
LANTERN_FROM_RAW_DECL(ScalarType, torch::ScalarType)
LANTERN_FROM_RAW_DECL(Scalar, torch::Scalar)
LANTERN_FROM_RAW_DECL(TensorOptions, torch::TensorOptions)
LANTERN_FROM_RAW_DECL(Device, torch::Device)
LANTERN_FROM_RAW_DECL(Dtype, torch::Dtype)
LANTERN_FROM_RAW_DECL(Dimname, torch::Dimname)
LANTERN_FROM_RAW_DECL(DimnameList, torch::DimnameList)
LANTERN_FROM_RAW_DECL(Generator, torch::Generator)
LANTERN_FROM_RAW_DECL(MemoryFormat, torch::MemoryFormat)
LANTERN_FROM_RAW_DECL(IntArrayRef, torch::IntArrayRef)
LANTERN_FROM_RAW_DECL(TensorDict, alias::TensorDict)
LANTERN_FROM_RAW_DECL(CompilationUnit, torch::CompilationUnit)
LANTERN_FROM_RAW_DECL(QScheme, torch::QScheme)
LANTERN_FROM_RAW_DECL(variable_list, torch::autograd::variable_list)
LANTERN_FROM_RAW_DECL(Layout, torch::Layout)
LANTERN_FROM_RAW_DECL(Storage, torch::Storage)
LANTERN_FROM_RAW_DECL(string, std::string)
LANTERN_FROM_RAW_DECL(string_view, c10::string_view)
LANTERN_FROM_RAW_DECL(int64_t, std::int64_t)
LANTERN_FROM_RAW_DECL(bool_t, bool)
LANTERN_FROM_RAW_DECL(double_t, double)
LANTERN_FROM_RAW_DECL(Stream, at::Stream)
LANTERN_FROM_RAW_DECL(IValue, torch::IValue)

namespace optional {
LANTERN_FROM_RAW_DECL(DimnameList, c10::optional<torch::DimnameList>)
LANTERN_FROM_RAW_DECL(Generator, c10::optional<torch::Generator>)
LANTERN_FROM_RAW_DECL(Tensor, c10::optional<torch::Tensor>)
LANTERN_FROM_RAW_DECL(double_t, c10::optional<double>)
LANTERN_FROM_RAW_DECL(int64_t, c10::optional<std::int64_t>)
LANTERN_FROM_RAW_DECL(bool_t, c10::optional<bool>)
LANTERN_FROM_RAW_DECL(ScalarType, c10::optional<torch::ScalarType>)
LANTERN_FROM_RAW_DECL(string, c10::optional<std::string>)
LANTERN_FROM_RAW_DECL(string_view, c10::optional<c10::string_view>)
LANTERN_FROM_RAW_DECL(MemoryFormat, c10::optional<torch::MemoryFormat>)
LANTERN_FROM_RAW_DECL(Scalar, c10::optional<torch::Scalar>)
LANTERN_FROM_RAW_DECL(TensorList, c10::List<c10::optional<torch::Tensor>>)
LANTERN_FROM_RAW_DECL(IntArrayRef, c10::optional<torch::IntArrayRef>)
LANTERN_FROM_RAW_DECL(DoubleArrayRef, c10::optional<torch::ArrayRef<double>>)
LANTERN_FROM_RAW_DECL(Device, c10::optional<torch::Device>)
}  // namespace optional

namespace vector {
LANTERN_FROM_RAW_DECL(string, std::vector<std::string>)
LANTERN_FROM_RAW_DECL(int64_t, std::vector<std::int64_t>)
LANTERN_FROM_RAW_DECL(double_t, std::vector<double>)
LANTERN_FROM_RAW_DECL(Scalar, std::vector<torch::Scalar>)
// This special type is used to allow converting to std::array<>'s
// of multiple different sizes.
LANTERN_FROM_RAW_DECL(bool_t, Vector<bool>)
}  // namespace vector

LANTERN_FROM_RAW_DECL(tuple, std::vector<void*>)
}  // namespace from_raw

// Types that are defined to allow owning the memory of all torch types.

struct NamedTupleHelper {
  std::vector<torch::IValue> elements;
  std::vector<std::string> names;
};

// a wrapper class for optional<torch::ArrayRef<T>> that owns all of it's memory
// and can easily be cast to the array ref type.
template <typename T>
class OptionalArrayRef {
 public:
  std::shared_ptr<std::vector<T>> x_;
  std::shared_ptr<c10::optional<torch::ArrayRef<T>>> x_ref_;
  OptionalArrayRef(const c10::optional<torch::ArrayRef<T>>& x) {
    if (x.has_value()) {
      x_ = std::make_shared<std::vector<T>>(x.value().vec());
      x_ref_ = std::make_shared<c10::optional<torch::ArrayRef<T>>>(*x_);
    } else {
      x_ref_ =
          std::make_shared<c10::optional<torch::ArrayRef<T>>>(c10::nullopt);
    }
  }
  OptionalArrayRef(const std::vector<T>& x) {
    if (x.size() == 0) {
      x_ref_ =
          std::make_shared<c10::optional<torch::ArrayRef<T>>>(c10::nullopt);
    } else {
      x_ = std::make_shared<std::vector<T>>(x);
      x_ref_ = std::make_shared<c10::optional<torch::ArrayRef<T>>>(*x_);
    }
  }
  operator c10::optional<torch::ArrayRef<T>> &() { return *x_ref_; }
};

template <typename Type>
class ArrayBox {
 public:
  std::shared_ptr<std::vector<Type>> buffer_;
  std::shared_ptr<torch::ArrayRef<Type>> x_;
  ArrayBox(const std::vector<Type>& x) {
    buffer_ = std::make_shared<std::vector<Type>>(x);
    x_ = std::make_shared<torch::ArrayRef<Type>>(*buffer_);
  }
  operator torch::ArrayRef<Type> &() { return *x_; }
  operator std::vector<Type> &() { return *buffer_; }
  void push_back(const Type& x) {
    buffer_->push_back(x);
    // We have to re-create the ArrayRef because the underlying buffer has
    // changed.
    x_ = std::make_shared<torch::ArrayRef<Type>>(*buffer_);
  }
};

template <typename T>
class Box {
 public:
  std::shared_ptr<T> x_;
  Box(const T& x) { x_ = std::make_shared<T>(x); }
  operator T&() { return *x_; }
};

// Objects return from lantern must own all memory necessary to re-use them.
// This is kind of easy for tensors as heap allocated tensors own all their
// memory memory. However this is not true for `torch::TensorList` which is just
// a a reference to a stack allocated vector of tensors. Thus if we simply
// return `torch::TensorList` we won't be able to reuse it when the stack
// allocated memory is no longer available.
//
// Types defined in this namespace are wrappers around torch types that don't
// own their memory thus need a wrapper that can hold the necessary information.
// Another important thing is that types here must be able to return their
// objects as references so they can be used in any possible lantern usage -
// that might include in-place modification of that object.
namespace self_contained {

using TensorList = ArrayBox<torch::Tensor>;
using Device = Box<torch::Device>;
using Dimname = Box<torch::Dimname>;
using DimnameList = ArrayBox<torch::Dimname>;
using IntArrayRef = ArrayBox<std::int64_t>;

class string_view {
 public:
  std::shared_ptr<std::string> s_;
  std::shared_ptr<c10::string_view> s_view_;
  string_view(const c10::string_view& x);
  operator c10::string_view &();
};

namespace vector {
using int64_t = ArrayBox<std::int64_t>;
}

namespace optional {

class DimnameList {
 public:
  std::shared_ptr<c10::optional<torch::DimnameList>> x_;
  std::shared_ptr<std::vector<torch::Dimname>> vec_;
  DimnameList(const c10::optional<torch::DimnameList>& x);
  operator c10::optional<torch::DimnameList> &();
};

class string_view {
 public:
  std::shared_ptr<std::string> s_;
  std::shared_ptr<c10::optional<c10::string_view>> s_view_;
  operator c10::optional<c10::string_view> &();
  string_view(const c10::optional<c10::string_view>& x);
};

using Generator = Box<c10::optional<torch::Generator>>;
using Tensor = Box<c10::optional<torch::Tensor>>;
using double_t = Box<c10::optional<double>>;
using int64_t = Box<c10::optional<std::int64_t>>;
using bool_t = Box<c10::optional<bool>>;
using ScalarType = Box<c10::optional<torch::ScalarType>>;
using string = Box<c10::optional<std::string>>;
using MemoryFormat = Box<c10::optional<torch::MemoryFormat>>;
using Scalar = Box<c10::optional<torch::Scalar>>;
using IntArrayRef = OptionalArrayRef<std::int64_t>;
using DoubleArrayRef = OptionalArrayRef<double>;
using Device = Box<c10::optional<torch::Device>>;

}  // namespace optional
}  // namespace self_contained

// Implementation of the functions defined earlier.
// Should only be included once in

#ifdef LANTERN_TYPES_IMPL

namespace self_contained {

string_view::string_view(const c10::string_view& x) {
  s_ = std::make_shared<std::string>(x.data(), x.size());
  s_view_ = std::make_shared<c10::string_view>(*s_);
}

string_view::operator c10::string_view &() { return *s_view_; }

namespace optional {

DimnameList::DimnameList(const c10::optional<torch::DimnameList>& x) {
  if (x.has_value()) {
    vec_ = std::make_shared<std::vector<torch::Dimname>>(x.value().vec());
    x_ = std::make_shared<c10::optional<torch::DimnameList>>(*vec_);
  } else {
    x_ = std::make_shared<c10::optional<torch::DimnameList>>(c10::nullopt);
  }
};

DimnameList::operator c10::optional<torch::DimnameList> &() { return *x_; };

string_view::string_view(const c10::optional<c10::string_view>& x) {
  if (x.has_value()) {
    s_ = std::make_shared<std::string>(x.value().data(), x.value().size());
    s_view_ = std::make_shared<c10::optional<c10::string_view>>(*s_);
  } else {
    s_view_ = std::make_shared<c10::optional<c10::string_view>>(c10::nullopt);
  }
};

string_view::operator c10::optional<c10::string_view> &() { return *s_view_; };

}  // namespace optional
}  // namespace self_contained

namespace make_raw {
void* Tensor(const torch::Tensor& x) { return make_ptr<torch::Tensor>(x); }
void* TensorList(const torch::TensorList& x) {
  return make_ptr<self_contained::TensorList>(x.vec());
}
void* ScalarType(const torch::ScalarType& x) {
  return make_ptr<torch::ScalarType>(x);
}
void* Scalar(const torch::Scalar& x) { return make_ptr<torch::Scalar>(x); }
void* TensorOptions(const torch::TensorOptions& x) {
  return make_ptr<torch::TensorOptions>(x);
}
void* Device(torch::Device& x) { return make_ptr<self_contained::Device>(x); }
void* Dtype(const torch::Dtype& x) { return make_ptr<torch::Dtype>(x); }
void* Dimname(torch::Dimname& x) {
  return make_ptr<self_contained::Dimname>(x);
}
void* DimnameList(const torch::DimnameList& x) {
  return make_ptr<self_contained::DimnameList>(x.vec());
}
void* Generator(const torch::Generator& x) {
  return make_ptr<torch::Generator>(x);
}
void* MemoryFormat(const torch::MemoryFormat& x) {
  return make_ptr<torch::MemoryFormat>(x);
}
void* IntArrayRef(const torch::IntArrayRef& x) {
  return make_ptr<self_contained::IntArrayRef>(x.vec());
}
void* TensorDict(const c10::Dict<std::string, torch::Tensor>& x) {
  return make_ptr<c10::Dict<std::string, torch::Tensor>>(x);
}
void* CompilationUnit(torch::jit::CompilationUnit& x) {
  return (void*)new torch::jit::CompilationUnit(std::move(x));
}
void* QScheme(const torch::QScheme& x) { return make_ptr<torch::QScheme>(x); }
void* variable_list(const torch::autograd::variable_list& x) {
  return make_ptr<torch::autograd::variable_list>(x);
}
void* Layout(const torch::Layout& x) { return make_ptr<torch::Layout>(x); }
void* Storage(const torch::Storage& x) { return make_ptr<torch::Storage>(x); }
void* string(const std::string& x) { return make_ptr<std::string>(x); }
void* string_view(const c10::string_view& x) {
  return make_ptr<self_contained::string_view>(x);
}
void* int64_t(const std::int64_t& x) { return make_ptr<std::int64_t>(x); }
void* double_t(const double& x) { return make_ptr<double>(x); }
void* bool_t(const bool& x) { return make_ptr<bool>(x); }
void* Stream(const at::Stream& x) { return make_ptr<at::Stream>(x); }
void* IValue(const at::IValue& x) { return make_ptr<at::IValue>(x); }

namespace vector {

void* string(const std::vector<std::string>& x) {
  return make_ptr<std::vector<std::string>>(x);
}
void* string() { return make_ptr<std::vector<std::string>>(); }
void* int64_t(const std::vector<std::int64_t>& x) {
  return make_ptr<self_contained::vector::int64_t>(x);
}
void* int64_t() { return make_ptr<std::vector<std::int64_t>>(); }
void* bool_t(const std::vector<bool>& x) { return make_ptr<Vector<bool>>(x); }
void* bool_t() { return make_ptr<Vector<bool>>(); }
void* double_t(const std::vector<double>& x) {
  return make_ptr<std::vector<double>>(x);
}
void* double_t() { return make_ptr<std::vector<double>>(); }
void* Scalar(const std::vector<torch::Scalar>& x) {
  return make_ptr<std::vector<torch::Scalar>>(x);
}

}  // namespace vector

namespace optional {

void* string(const c10::optional<std::string>& x) {
  return make_ptr<self_contained::optional::string>(x);
}

void* string_view(const c10::optional<c10::string_view>& x) {
  return make_ptr<self_contained::optional::string_view>(x);
}

void* TensorList(const c10::List<c10::optional<torch::Tensor>>& x) {
  return make_ptr<c10::List<c10::optional<torch::Tensor>>>(x);
}

void* IntArrayRef(const c10::optional<torch::ArrayRef<std::int64_t>>& x) {
  return make_ptr<OptionalArrayRef<std::int64_t>>(x);
}

void* DoubleArrayRef(const c10::optional<torch::ArrayRef<double>>& x) {
  return make_ptr<OptionalArrayRef<double>>(x);
}

void* Scalar(const c10::optional<torch::Scalar>& x) {
  return make_ptr<self_contained::optional::Scalar>(x);
}

void* DimnameList(const c10::optional<torch::DimnameList>& x) {
  return make_ptr<self_contained::optional::DimnameList>(x);
}

void* Generator(const c10::optional<torch::Generator>& x) {
  return make_ptr<self_contained::optional::Generator>(x);
}

void* Tensor(const c10::optional<torch::Tensor>& x) {
  return make_ptr<self_contained::optional::Tensor>(x);
}

void* double_t(const c10::optional<double>& x) {
  return make_ptr<self_contained::optional::double_t>(x);
}

void* int64_t(const c10::optional<std::int64_t>& x) {
  return make_ptr<self_contained::optional::int64_t>(x);
}

void* bool_t(const c10::optional<bool>& x) {
  return make_ptr<self_contained::optional::bool_t>(x);
}

void* ScalarType(const c10::optional<torch::ScalarType>& x) {
  return make_ptr<self_contained::optional::ScalarType>(x);
}

void* MemoryFormat(const c10::optional<torch::MemoryFormat>& x) {
  return make_ptr<self_contained::optional::MemoryFormat>(x);
}

void* Device(const c10::optional<torch::Device>& x) {
  return make_ptr<self_contained::optional::Device>(x);
}

}  // namespace optional

}  // namespace make_raw

#define LANTERN_FROM_RAW(name, type) \
  type& name(void* x) { return *reinterpret_cast<type*>(x); }

#define LANTERN_FROM_RAW_WRAPPED(name, wraper_type, type) \
  type& name(void* x) { return *reinterpret_cast<wraper_type*>(x); }

namespace alias {
using TensorDict = c10::Dict<std::string, torch::Tensor>;
}

namespace from_raw {
LANTERN_FROM_RAW(Tensor, torch::Tensor)
LANTERN_FROM_RAW_WRAPPED(TensorList, self_contained::TensorList,
                         torch::TensorList)
LANTERN_FROM_RAW(ScalarType, torch::ScalarType)
LANTERN_FROM_RAW(Scalar, torch::Scalar)
LANTERN_FROM_RAW(TensorOptions, torch::TensorOptions)
LANTERN_FROM_RAW_WRAPPED(Device, self_contained::Device, torch::Device)
LANTERN_FROM_RAW(Dtype, torch::Dtype)
LANTERN_FROM_RAW_WRAPPED(Dimname, self_contained::Dimname, torch::Dimname)
LANTERN_FROM_RAW_WRAPPED(DimnameList, self_contained::DimnameList,
                         torch::DimnameList)
LANTERN_FROM_RAW(Generator, torch::Generator)
LANTERN_FROM_RAW(MemoryFormat, torch::MemoryFormat)
LANTERN_FROM_RAW_WRAPPED(IntArrayRef, self_contained::IntArrayRef,
                         torch::IntArrayRef)
LANTERN_FROM_RAW(TensorDict, alias::TensorDict)
LANTERN_FROM_RAW(CompilationUnit, torch::jit::CompilationUnit)
LANTERN_FROM_RAW(QScheme, torch::QScheme)
LANTERN_FROM_RAW(variable_list, torch::autograd::variable_list)
LANTERN_FROM_RAW(Layout, torch::Layout)
LANTERN_FROM_RAW(Storage, torch::Storage)
LANTERN_FROM_RAW(string, std::string)
LANTERN_FROM_RAW_WRAPPED(string_view, self_contained::string_view,
                         c10::string_view)
LANTERN_FROM_RAW(int64_t, std::int64_t)
LANTERN_FROM_RAW(bool_t, bool)
LANTERN_FROM_RAW(double_t, double)
LANTERN_FROM_RAW(Stream, at::Stream)
LANTERN_FROM_RAW(IValue, torch::IValue)

namespace optional {
LANTERN_FROM_RAW_WRAPPED(DimnameList, self_contained::optional::DimnameList,
                         c10::optional<torch::DimnameList>)
LANTERN_FROM_RAW_WRAPPED(Generator, self_contained::optional::Generator,
                         c10::optional<torch::Generator>)
LANTERN_FROM_RAW_WRAPPED(Tensor, self_contained::optional::Tensor,
                         c10::optional<torch::Tensor>)
LANTERN_FROM_RAW_WRAPPED(double_t, self_contained::optional::double_t,
                         c10::optional<double>)
LANTERN_FROM_RAW_WRAPPED(int64_t, self_contained::optional::int64_t,
                         c10::optional<std::int64_t>)
LANTERN_FROM_RAW_WRAPPED(bool_t, self_contained::optional::bool_t,
                         c10::optional<bool>)
LANTERN_FROM_RAW_WRAPPED(ScalarType, self_contained::optional::ScalarType,
                         c10::optional<torch::ScalarType>)
LANTERN_FROM_RAW_WRAPPED(string, self_contained::optional::string,
                         c10::optional<std::string>)
LANTERN_FROM_RAW_WRAPPED(string_view, self_contained::optional::string_view,
                         c10::optional<c10::string_view>)
LANTERN_FROM_RAW_WRAPPED(MemoryFormat, self_contained::optional::MemoryFormat,
                         c10::optional<torch::MemoryFormat>)
LANTERN_FROM_RAW_WRAPPED(Scalar, self_contained::optional::Scalar,
                         c10::optional<torch::Scalar>)
LANTERN_FROM_RAW(TensorList, c10::List<c10::optional<torch::Tensor>>)
LANTERN_FROM_RAW_WRAPPED(IntArrayRef, self_contained::optional::IntArrayRef,
                         c10::optional<torch::IntArrayRef>)
LANTERN_FROM_RAW_WRAPPED(DoubleArrayRef,
                         self_contained::optional::DoubleArrayRef,
                         c10::optional<torch::ArrayRef<double>>)
LANTERN_FROM_RAW_WRAPPED(Device, self_contained::optional::Device,
                         c10::optional<torch::Device>)
}  // namespace optional

namespace vector {
LANTERN_FROM_RAW(string, std::vector<std::string>)
LANTERN_FROM_RAW_WRAPPED(int64_t, self_contained::vector::int64_t,
                         std::vector<std::int64_t>)
LANTERN_FROM_RAW(bool_t, Vector<bool>)
LANTERN_FROM_RAW(double_t, std::vector<double>)
LANTERN_FROM_RAW(Scalar, std::vector<torch::Scalar>)
}  // namespace vector

LANTERN_FROM_RAW(tuple, std::vector<void*>)
}  // namespace from_raw

#endif  // LANTERN_TYPES_IMPL