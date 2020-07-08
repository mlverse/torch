#include <iostream>

#define LANTERN_BUILD

#include "lantern/lantern.h"

#include <torch/torch.h>

#include "utils.hpp"

void *_lantern_MemoryFormat_Contiguous()
{
  LANTERN_FUNCTION_START
  return (void *)new LanternObject<torch::MemoryFormat>(torch::MemoryFormat::Contiguous);
  LANTERN_FUNCTION_END
}

void *_lantern_MemoryFormat_Preserve()
{
  LANTERN_FUNCTION_START
  return (void *)new LanternObject<torch::MemoryFormat>(torch::MemoryFormat::Preserve);
  LANTERN_FUNCTION_END
}

void *_lantern_MemoryFormat_ChannelsLast()
{
  LANTERN_FUNCTION_START
  return (void *)new LanternObject<torch::MemoryFormat>(torch::MemoryFormat::ChannelsLast);
  LANTERN_FUNCTION_END
}

const char *_lantern_MemoryFormat_type(void *format)
{
  LANTERN_FUNCTION_START
  torch::MemoryFormat y = reinterpret_cast<LanternObject<torch::MemoryFormat> *>(format)->get();

  if (y == torch::MemoryFormat::Contiguous)
  {
    return "contiguous";
  }

  if (y == torch::MemoryFormat::Preserve)
  {
    return "preserve";
  }

  if (y == torch::MemoryFormat::ChannelsLast)
  {
    return "channels_last";
  }

  return "undefined";
  LANTERN_FUNCTION_END
}
