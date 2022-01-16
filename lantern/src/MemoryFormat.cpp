#include <iostream>

#define LANTERN_BUILD

#include <torch/torch.h>

#include "lantern/lantern.h"
#include "utils.hpp"

void *_lantern_MemoryFormat_Contiguous() {
  LANTERN_FUNCTION_START
  return make_raw::MemoryFormat(torch::MemoryFormat::Contiguous);
  LANTERN_FUNCTION_END
}

void *_lantern_MemoryFormat_Preserve() {
  LANTERN_FUNCTION_START
  return make_raw::MemoryFormat(torch::MemoryFormat::Preserve);
  LANTERN_FUNCTION_END
}

void *_lantern_MemoryFormat_ChannelsLast() {
  LANTERN_FUNCTION_START
  return make_raw::MemoryFormat(torch::MemoryFormat::ChannelsLast);
  LANTERN_FUNCTION_END
}

const char *_lantern_MemoryFormat_type(void *format) {
  LANTERN_FUNCTION_START
  auto y = from_raw::MemoryFormat(format);

  std::string str;
  if (y == torch::MemoryFormat::Contiguous) {
    str = "contiguous";
  } else if (y == torch::MemoryFormat::Preserve) {
    str = "preserve";
  } else if (y == torch::MemoryFormat::ChannelsLast) {
    str = "channels_last";
  } else {
    str = "undefined";
  }

  char *cstr = new char[str.length() + 1];
  strcpy(cstr, str.c_str());
  return cstr;
  LANTERN_FUNCTION_END
}
