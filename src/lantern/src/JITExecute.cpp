#define LANTERN_BUILD
#include <torch/torch.h>
#include "lantern/lantern.h"
#include "lantern/types.h"

void* _lantern_function_schema_name(void* schema) {
  LANTERN_FUNCTION_START
  const auto& s {from_raw::FunctionSchema(schema)};
  const auto& name {s.name()};
  return make_raw::string(name);
  LANTERN_FUNCTION_END
}

void* _lantern_function_schema_arguments(void* schema) {
  LANTERN_FUNCTION_START
  const auto& s {from_raw::FunctionSchema(schema)};
  const auto& arguments {s.arguments()};
  return make_raw::vector::Argument(arguments);
  LANTERN_FUNCTION_END
}

int _lantern_function_schema_num_arguments(void* arglist) {
  LANTERN_FUNCTION_START
  const auto& a {from_raw::vector::Argument(arglist)};
  return a.size();
  LANTERN_FUNCTION_END
}

void* _lantern_function_schema_argument_at(void* arglist, int i) {
  LANTERN_FUNCTION_START
  const auto& a {from_raw::vector::Argument(arglist)};
  const auto& at {a.at(i)};
  return make_raw::Argument(at);
  LANTERN_FUNCTION_END
}

void* _lantern_function_schema_argument_name(void* argument) {
  LANTERN_FUNCTION_START
  const auto& a {from_raw::Argument(argument)};
  const auto& argname {a.name()};
  return make_raw::string(argname);
  LANTERN_FUNCTION_END
}

void* _lantern_function_schema_argument_type(void* argument) {
  LANTERN_FUNCTION_START
  const auto& a {from_raw::Argument(argument)};
  const auto& argtype {c10::typeKindToString(a.type()->kind())};
  return make_raw::string(argtype);
  LANTERN_FUNCTION_END
}

void* _lantern_function_schema_returns(void* schema) {
  LANTERN_FUNCTION_START
  const auto& s {from_raw::FunctionSchema(schema)};
  const auto& returns {s.returns()};
  return make_raw::vector::Argument(returns);
  LANTERN_FUNCTION_END
}

void* _lantern_function_schema_return_type(void* return_) {
  LANTERN_FUNCTION_START
  const auto& r {from_raw::Argument(return_)};
  const auto& argtype {c10::typeKindToString(r.type()->kind())};
  return make_raw::string(argtype);
  LANTERN_FUNCTION_END
}

int _lantern_function_schema_list_size(void* schema_list) {
  LANTERN_FUNCTION_START
  const auto& s {from_raw::vector::FunctionSchema(schema_list)};
  return s.size();
  LANTERN_FUNCTION_END
}

void* _lantern_function_schema_list_at(void* schema_list, int i) {
  LANTERN_FUNCTION_START
  const auto& a {from_raw::vector::FunctionSchema(schema_list)};
  const auto& at {a.at(i)};
  return make_raw::FunctionSchema(at);
  LANTERN_FUNCTION_END
}

