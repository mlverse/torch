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