#include <iostream>

#define LANTERN_BUILD

#include "lantern/lantern.h"

#include <torch/torch.h>

#include "utils.hpp"

void *lantern_Dimname(const char *name)
{
  auto nm = torch::Dimname::fromSymbol(torch::Symbol::dimname(std::string(name)));
  return (void *)new LanternPtr<torch::Dimname>(nm);
}

void *lantern_DimnameList()
{
  return (void *)new LanternPtr<std::vector<torch::Dimname>>();
}

void lantern_DimnameList_push_back(void *list, void *dimname)
{
  torch::Dimname nm = reinterpret_cast<LanternPtr<torch::Dimname> *>(dimname)->get();
  reinterpret_cast<LanternPtr<std::vector<torch::Dimname>> *>(list)->get().push_back(nm);
}

void *lantern_DimnameList_at(void *list, int i)
{
  torch::Dimname x = reinterpret_cast<LanternPtr<std::vector<torch::Dimname>> *>(list)->get().at(i);
  return (void *)new LanternPtr<torch::Dimname>(x);
}

int64_t lantern_DimnameList_size(void *list)
{
  return reinterpret_cast<LanternPtr<std::vector<torch::Dimname>> *>(list)->get().size();
}

const char *lantern_Dimname_to_string(void *dimname)
{
  torch::Dimname nm = reinterpret_cast<LanternPtr<torch::Dimname> *>(dimname)->get();
  return nm.symbol().toUnqualString();
}
