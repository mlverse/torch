#define LANTERN_BUILD
#include <torch/torch.h>
#include "lantern/lantern.h"
#include "lantern/types.h"

void* _lantern_function_schema_name(void* schema) {
  LANTERN_FUNCTION_START
  const auto& s {from_raw::FunctionSchema(schema)};
  const auto& name {s.name()};
  return make_raw::string(name);
  LANTERN_FUNCTION_END_RET(nullptr)
}

void* _lantern_function_schema_arguments(void* schema) {
  LANTERN_FUNCTION_START
  const auto& s {from_raw::FunctionSchema(schema)};
  const auto& arguments {s.arguments()};
  return make_raw::vector::Argument(arguments);
  LANTERN_FUNCTION_END_RET(nullptr)
}

int _lantern_function_schema_num_arguments(void* arglist) {
  LANTERN_FUNCTION_START
  const auto& a {from_raw::vector::Argument(arglist)};
  return a.size();
  LANTERN_FUNCTION_END_RET(0)
}

void* _lantern_function_schema_argument_at(void* arglist, int i) {
  LANTERN_FUNCTION_START
  const auto& a {from_raw::vector::Argument(arglist)};
  const auto& at {a.at(i)};
  return make_raw::Argument(at);
  LANTERN_FUNCTION_END_RET(nullptr)
}

void* _lantern_function_schema_argument_name(void* argument) {
  LANTERN_FUNCTION_START
  const auto& a {from_raw::Argument(argument)};
  const auto& argname {a.name()};
  return make_raw::string(argname);
  LANTERN_FUNCTION_END_RET(nullptr)
}

void* _lantern_function_schema_argument_type(void* argument) {
  LANTERN_FUNCTION_START
  const auto& a {from_raw::Argument(argument)};
  const auto& argtype {c10::typeKindToString(a.type()->kind())};
  return make_raw::string(argtype);
  LANTERN_FUNCTION_END_RET(nullptr)
}

void* _lantern_function_schema_returns(void* schema) {
  LANTERN_FUNCTION_START
  const auto& s {from_raw::FunctionSchema(schema)};
  const auto& returns {s.returns()};
  return make_raw::vector::Argument(returns);
  LANTERN_FUNCTION_END_RET(nullptr)
}

void* _lantern_function_schema_return_type(void* return_) {
  LANTERN_FUNCTION_START
  const auto& r {from_raw::Argument(return_)};
  const auto& argtype {c10::typeKindToString(r.type()->kind())};
  return make_raw::string(argtype);
  LANTERN_FUNCTION_END_RET(nullptr)
}

int _lantern_function_schema_list_size(void* schema_list) {
  LANTERN_FUNCTION_START
  const auto& s {from_raw::vector::FunctionSchema(schema_list)};
  return s.size();
  LANTERN_FUNCTION_END_RET(0)
}

void* _lantern_function_schema_list_at(void* schema_list, int i) {
  LANTERN_FUNCTION_START
  const auto& a {from_raw::vector::FunctionSchema(schema_list)};
  const auto& at {a.at(i)};
  return make_raw::FunctionSchema(at);
  LANTERN_FUNCTION_END_RET(nullptr)
}

// Return a list of registered operators.
void* _lantern_jit_all_operators() {
  LANTERN_FUNCTION_START
  const auto& ops {torch::jit::getAllOperators()};
  std::vector<std::string> names;
  for (const auto& op : ops) {
    names.push_back(op->schema().name());
  }
  return make_raw::vector::string(names);
  LANTERN_FUNCTION_END_RET(nullptr)
}

// Obtain the default FunctionSchema for a specific operator.
// The schema contains information about arguments (name and type) and return types.
void* _lantern_jit_operator_info(void* name) {
  LANTERN_FUNCTION_START
  const auto& op_name {c10::Symbol::fromQualString(from_raw::string(name))};
  const auto& ops {torch::jit::getAllOperatorsFor(op_name)};
  const auto length {ops.size()};
  const auto& op {ops.front()};
  return make_raw::FunctionSchema(op->schema());
  LANTERN_FUNCTION_END_RET(nullptr)
}

// Returns all FunctionSchemas for a specific operator.
void* _lantern_jit_all_schemas_for(void* name) {
  LANTERN_FUNCTION_START
  const auto& op_name {c10::Symbol::fromQualString(from_raw::string(name))};
  const auto& ops {torch::jit::getAllOperatorsFor(op_name)};
  std::vector<c10::FunctionSchema> schemas;
  for (const auto& op : ops) {
    schemas.push_back(op->schema());
  }
  return make_raw::vector::FunctionSchema(schemas);
  LANTERN_FUNCTION_END_RET(nullptr)
}

// Executes the requested operation using the first FunctionSchema that matches the given arguments.
void* _lantern_jit_execute(void* name, void* stack) {
  LANTERN_FUNCTION_START
  const auto& op_name {c10::Symbol::fromQualString(from_raw::string(name))};
  const auto& ops {torch::jit::getAllOperatorsFor(op_name)};
  const auto length {ops.size()};
  auto& stack_ {*reinterpret_cast<torch::jit::Stack*>(stack)};
  if (length <= 0) {
    throw std::runtime_error("Operator not found.");
  } else if (length == 1) {
    const auto& op {ops.front()};
    op->getOperation()(stack_);
  } else {
    LLOG("Found several matches; looking for first viable schema.\n")
    int tries {0};
    bool found {false};
    for (const auto& op : ops) {
      const auto& schema {op->schema()};
      const auto& arguments {schema.arguments()};
      const auto num_args {arguments.size()};
      const auto num_given {stack_.size()};
      if (num_given != num_args) {
        LLOG("Schema skipped: Number of arguments given does not match expected.\n")
        tries++;
        continue;
      }
      try {
        op->getOperation()(stack_);
        tries++;
        found = true;
        break;
      } catch (c10::Error e) {
        const auto& schema {op->schema()};
        const auto& arguments {schema.arguments()};
        const auto& returns {schema.returns()};
        std::string info{"Couldn't call operator, schema is:\n"};
        info += "Arguments:\n";
        for (const auto& arg : arguments) {
          info += "  Name: " + std::string(arg.name()) + "\n";
          info += "  Type: " + std::string(c10::typeKindToString(arg.type()->kind())) + "\n";
        }
        info += "Returns:\n";
        for (const auto& ret : returns) {
          info += "  Type: " + std::string(c10::typeKindToString(ret.type()->kind())) + "\n";
        }
        info += "C10 error is: " + std::string(e.what_without_backtrace());
        info += "\nTrying next.\n";
        LLOG(info.c_str())
      }
    }
  if (found) {
    LLOG(("Found matching schema in try: " + std::to_string(tries)).c_str())
  } else {
    LLOG("Tried all schemas; none matched.\n")
    throw std::runtime_error("Tried all schemas; none matched.");
  }
  }
  return (void*)std::make_unique<torch::jit::Stack>(stack_).release();
  LANTERN_FUNCTION_END_RET(nullptr)
}
