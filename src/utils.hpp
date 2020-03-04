#pragma once

template <class T>
class LanternObject
{
private:
  T _object;

public:
  LanternObject(T object)
  {
    _object = object;
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