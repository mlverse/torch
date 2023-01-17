#include <iostream>

#define LANTERN_BUILD

#include <torch/torch.h>

#include "lantern/lantern.h"
#include "utils.hpp"

void *_lantern_Dimname(void *name) {
  LANTERN_FUNCTION_START
  auto name_ = from_raw::string(name);
  auto nm = torch::Dimname::fromSymbol(torch::Symbol::dimname(name_));
  return make_raw::Dimname(nm);
  LANTERN_FUNCTION_END
}

void *_lantern_DimnameList() {
  LANTERN_FUNCTION_START
  return make_raw::DimnameList({});
  LANTERN_FUNCTION_END
}

void _lantern_DimnameList_push_back(void *list, void *dimname) {
  LANTERN_FUNCTION_START
  torch::Dimname nm = from_raw::Dimname(dimname);
  // Extending a DimnameList is not allowed. Thus we need to extend the buffer
  // and re-create the ArrayRef.
  reinterpret_cast<self_contained::DimnameList *>(list)->push_back(nm);
  LANTERN_FUNCTION_END_VOID
}

void *_lantern_DimnameList_at(void *list, int i) {
  LANTERN_FUNCTION_START
  torch::Dimname x = from_raw::DimnameList(list).at(i);
  return make_raw::Dimname(x);
  LANTERN_FUNCTION_END
}

int64_t _lantern_DimnameList_size(void *list) {
  LANTERN_FUNCTION_START
  return from_raw::DimnameList(list).size();
  LANTERN_FUNCTION_END_RET(0)
}

const char *_lantern_Dimname_to_string(void *dimname) {
  LANTERN_FUNCTION_START
  auto nm = from_raw::Dimname(dimname);
  std::string str = nm.symbol().toUnqualString();
  char *cstr = new char[str.length() + 1];
  strcpy(cstr, str.c_str());
  return cstr;
  LANTERN_FUNCTION_END
}
