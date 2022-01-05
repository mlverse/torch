#pragma once
#include "lantern/types.h"

template <class T>
class LanternPtr
{
private:
  T *_object;

public:
  LanternPtr(const T &object)
  {
    _object = new T(object);
  }

  LanternPtr()
  {
    _object = new T;
  }

  ~LanternPtr()
  {
    delete _object;
    _object = NULL;
  }

  T &get()
  {
    return *_object;
  }
  
  operator T &() const
  {
    return *_object;
  }

};

struct NamedTupleHelper {
    std::vector<torch::IValue> elements;
    std::vector<std::string> names;
};

// a wrapper class for optional<torch::ArrayRef<T>> that owns all of it's memory and
// can easily be cast to the array ref type.
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
        x_ref_ = std::make_shared<c10::optional<torch::ArrayRef<T>>>(c10::nullopt);
      }
    }
    OptionalArrayRef(const std::vector<T>& x){
      if (x.size() == 0) {
        x_ref_ = std::make_shared<c10::optional<torch::ArrayRef<T>>>(c10::nullopt);
      } else {
        x_ = std::make_shared<std::vector<T>>(x);
        x_ref_ = std::make_shared<c10::optional<torch::ArrayRef<T>>>(*x_);
      }
    }
    operator c10::optional<torch::ArrayRef<T>> &() {
      return *x_ref_;
    }
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
    operator torch::ArrayRef<Type>&() {
      return *x_;
    }
    operator std::vector<Type>&() {
      return *buffer_;
    }
    void push_back (const Type& x) {
      buffer_->push_back(x);
      // We have to re-create the ArrayRef because the underlying buffer has changed.
      x_ = std::make_shared<torch::ArrayRef<Type>>(*buffer_);
    }
};

template <typename T>
class Box {
  public:
    std::shared_ptr<T> x_;
    Box(const T& x) {
      x_ = std::make_shared<T>(x);
    }
    operator T&() {
      return *x_;
    }
};

// Objects return from lantern must own all memory necessary to re-use them.
// This is kind of easy for tensors as heap allocated tensors own all their memory
// memory. However this is not true for `torch::TensorList` which is just a a reference
// to a stack allocated vector of tensors. Thus if we simply return `torch::TensorList`
// we won't be able to reuse it when the stack allocated memory is no longer available.
// 
// Types defined in this namespace are wrappers around torch types that don't own their memory
// thus need a wrapper that can hold the necessary information.
// Another important thing is that types here must be able to return their objects as references
// so they can be used in any possible lantern usage - that might include in-place modification
// of that object.
namespace self_contained {

  using TensorList = ArrayBox<torch::Tensor>;
  using Device = Box<torch::Device>;
  using Dimname = Box<torch::Dimname>;
  using DimnameList = ArrayBox<torch::Dimname>;
  using IntArrayRef = ArrayBox<std::int64_t>;

  namespace vector {
    using int64_t = ArrayBox<std::int64_t>;
  }

  namespace optional {
    
    class DimnameList {
      public:
        std::shared_ptr<c10::optional<torch::DimnameList>> x_;
        std::shared_ptr<std::vector<torch::Dimname>> vec_;
        DimnameList (const c10::optional<torch::DimnameList>& x);
        operator c10::optional<torch::DimnameList>&();
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
    
  }
}



