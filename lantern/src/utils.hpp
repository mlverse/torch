#pragma once

template <class T>
class LanternObject
{
private:
  T _object;

public:
  LanternObject(T object) : _object(std::forward<T>(object))
  {
  }

  LanternObject()
  {
  }

  T &get()
  {
    return _object;
  }
};

template <class T>
class LanternPtr
{
private:
  T *_object;

public:
  LanternPtr(T &object)
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
};

// https://pt.stackoverflow.com/a/438284/6036
//
template <class... T>
std::vector<void *> to_vector(std::tuple<T...> x)
{
  std::vector<void *> out;
  out.reserve(sizeof...(T));
  std::apply([&out](auto &&... args) {
    ((out.push_back(new std::remove_reference_t<decltype(args)>(std::forward<decltype(args)>(args)))), ...);
  },
             x);
  return out;
}

template <class T>
auto optional(void *x)
{

  if (x == nullptr)
  {
    return std::make_shared<LanternObject<c10::optional<T>>>(c10::nullopt);
  } 

  auto z = ((LanternObject<T> *)x)->get();
  return std::make_shared<LanternObject<c10::optional<T>>>(z);
}
