#include <iostream>

#define LANTERN_BUILD

#include "lantern/lantern.h"

#include <torch/torch.h>

#include "utils.hpp"

void *lantern_MemoryFormat_Contiguous()
{
  return (void *)new LanternObject<torch::MemoryFormat>(torch::MemoryFormat::Contiguous);
}

void *lantern_MemoryFormat_Preserve()
{
  return (void *)new LanternObject<torch::MemoryFormat>(torch::MemoryFormat::Preserve);
}

void *lantern_MemoryFormat_ChannelsLast()
{
  return (void *)new LanternObject<torch::MemoryFormat>(torch::MemoryFormat::ChannelsLast);
}

const char *lantern_MemoryFormat_type(void *format)
{

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
}
