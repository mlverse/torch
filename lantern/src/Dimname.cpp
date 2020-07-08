#include <iostream>

#define LANTERN_BUILD

#include "lantern/lantern.h"

#include <torch/torch.h>

#include "utils.hpp"

void *_lantern_Dimname(const char *name)
{
  LANTERN_FUNCTION_START
  auto nm = torch::Dimname::fromSymbol(torch::Symbol::dimname(std::string(name)));
  return (void *)new LanternPtr<torch::Dimname>(nm);
  LANTERN_FUNCTION_END
}

void *_lantern_DimnameList()
{
  LANTERN_FUNCTION_START
  return (void *)new LanternPtr<std::vector<torch::Dimname>>();
  LANTERN_FUNCTION_END
}

void _lantern_DimnameList_push_back(void *list, void *dimname)
{
  LANTERN_FUNCTION_START
  torch::Dimname nm = reinterpret_cast<LanternPtr<torch::Dimname> *>(dimname)->get();
  reinterpret_cast<LanternPtr<std::vector<torch::Dimname>> *>(list)->get().push_back(nm);
  LANTERN_FUNCTION_END_VOID
}

void *_lantern_DimnameList_at(void *list, int i)
{
  LANTERN_FUNCTION_START
  torch::Dimname x = reinterpret_cast<LanternPtr<std::vector<torch::Dimname>> *>(list)->get().at(i);
  return (void *)new LanternPtr<torch::Dimname>(x);
  LANTERN_FUNCTION_END
}

int64_t _lantern_DimnameList_size(void *list)
{
  LANTERN_FUNCTION_START
  return reinterpret_cast<LanternPtr<std::vector<torch::Dimname>> *>(list)->get().size();
  LANTERN_FUNCTION_END_RET(0)
}

const char *_lantern_Dimname_to_string(void *dimname)
{
  LANTERN_FUNCTION_START
  torch::Dimname nm = reinterpret_cast<LanternPtr<torch::Dimname> *>(dimname)->get();
  return nm.symbol().toUnqualString();
  LANTERN_FUNCTION_END
}
