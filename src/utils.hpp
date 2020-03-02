#pragma once

template<class T>
class LanternObject
{
private:
  T _object;
public:
  LanternObject(T object)
  {
    _object = object;
  }
  
  T& get()
  {
    return _object;
  }
};

template<class T>
class LanternPtr
{
private:
  T* _object;
public:
  LanternPtr(T& object)
  {
    _object = new T(object);
  }
  
  ~LanternPtr()
  {
    delete _object;
    _object = NULL;
  }
  
  T& get()
  {
    return *_object;
  }
};