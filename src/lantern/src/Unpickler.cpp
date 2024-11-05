#include <ATen/ATen.h>
#include <ATen/core/Dict.h>
#include <ATen/core/jit_type.h>
#include <ATen/core/jit_type_base.h>
#include <c10/util/Optional.h>
#include <torch/csrc/jit/ir/ir.h>
#define private public
#include "Unpickler.h"
#undef private
#include <torch/csrc/jit/api/function_impl.h>
#include <torch/csrc/jit/mobile/type_parser.h>
#include <torch/csrc/jit/serialization/pickler.h>
#include <torch/csrc/jit/serialization/storage_context.h>
#include <torch/csrc/jit/serialization/unpickler.h>
#include <torch/csrc/utils/byte_order.h>
#include <string>
#include <utility>

namespace torch {
namespace jit {

static void restoreAccurateTypeTagsIfPossible(const IValue& root) {
  if (root.isObject()) {
    restoreAccurateTypeTags(root, root.type());
  }
}

// Pickled objects are stored in a form compatible with Python pickling.
// In torchscript List[T]/Dict[K, V] are statically typed and contain
// dynamic type tags that allow T, K, and V to be recovered. But this
// info is not stored in the Python pickling information. However, we
// can recover this information from the static type of the top-level
// object being unpickled, because we have a record of the type of the
// objects it contains as attributes.
// `IfPossible` - we can only do this recovery when we have an object as
// the top-level unpickled thing (which is guaranteed for Modules, but
// not for torch.load/torch.save). Otherwise we do not know the types
// of the contained objects and cannot restore the tags.
void restoreAccurateTypeTags(const IValue& root, const TypePtr& type_tag) {
  struct Work {
    TypePtr type;
    IValue value;
  };
  std::vector<Work> to_process = {{type_tag, root}};
  std::unordered_set<const void*> scanned;
  while (!to_process.empty()) {
    Work w = std::move(to_process.back());
    to_process.pop_back();
    // ensure we only scan each pointer value once, otherwise this
    // can become exponential (and if we allow recursive data in the future,
    // it would not terminiate).
    if (w.value.isPtrType()) {
      const void* key = w.value.internalToPointer();
      auto it = scanned.find(key);
      if (it != scanned.end()) {
        continue;
      }
      scanned.emplace_hint(it, key);
    }
    auto kind = w.type->kind();
    if (auto dyn = w.type->castRaw<c10::DynamicType>()) {
      kind = dyn->dynamicKind();
    }
    switch (kind) {
      case TensorType::Kind:
      case StorageType::Kind:
      case NumberType::Kind:
      case FloatType::Kind:
      case ComplexType::Kind:
      case IntType::Kind:
      case NoneType::Kind:
      case GeneratorType::Kind:
      case QuantizerType::Kind:
      case BoolType::Kind:
      case VarType::Kind:
      case CapsuleType::Kind:
      case PyObjectType::Kind:
      case StringType::Kind:
      case FunctionType::Kind:
      case DeviceObjType::Kind:
      case StreamObjType::Kind:
      case QSchemeType::Kind:
      case LayoutType::Kind:
      case MemoryFormatType::Kind:
      case ScalarTypeType::Kind:
      case RRefType::Kind:
      case AnyType::Kind:
      case AnyListType::Kind:
      case AnyTupleType::Kind:
      case AnyClassType::Kind:
      case AnyEnumType::Kind:
        // no op, there is nothing to tag
        break;
      case c10::SymIntType::Kind:
        // TODO: Can this really show up though? :think:
        TORCH_CHECK(!w.value.toSymInt().is_heap_allocated());
        // no op, there is nothing to tag
        break;
      case c10::SymFloatType::Kind:
        TORCH_CHECK(!w.value.toSymFloat().is_symbolic());
        // no op, there is nothing to tag
        break;
      case c10::SymBoolType::Kind:
        TORCH_CHECK(!w.value.toSymBool().is_heap_allocated());
        // no op, there is nothing to tag
        break;
      case DynamicType::Kind:
      case UnionType::Kind:
      case EnumType::Kind:
        // TODO(gmagogsfm): Implement serialization/deserialization of Enum.
        TORCH_INTERNAL_ASSERT(false);
      case TupleType::Kind: {
        auto t = w.value.toTuple();
        for (size_t i = 0; i < w.type->containedTypeSize(); ++i) {
          Work elem = {w.type->containedType(i), t->elements().at(i)};
          to_process.emplace_back(std::move(elem));
        }
      } break;
      case FutureType::Kind: {
        auto f = w.value.toFuture();
        if (f->completed()) {
          Work elem = {w.type->containedType(0), f->value()};
          to_process.emplace_back(std::move(elem));
        }
      } break;
      case AwaitType::Kind: {
        auto aw = w.value.toAwait();
        if (aw->completed()) {
          Work elem = {w.type->containedType(0), aw->wait()};
          to_process.emplace_back(std::move(elem));
        }
      } break;
      case OptionalType::Kind: {
        if (!w.value.isNone()) {
          Work elem = {w.type->containedType(0), w.value};
          to_process.emplace_back(std::move(elem));
        }
      } break;
      case ListType::Kind: {
        // specialized lists do not need their type refined, so we can exit
        // early here
        if (!w.value.isList()) {
          break;
        }
        auto elem_type = w.type->containedType(0);
        auto lst = w.value.toList();
        lst.unsafeSetElementType(elem_type);
        for (const IValue& item : lst) {
          Work elem = {elem_type, item};
          to_process.emplace_back(std::move(elem));
        }
      } break;
      case DictType::Kind: {
        auto d = w.value.toGenericDict();
        auto keyType = w.type->containedType(0);
        auto valType = w.type->containedType(1);
        d.unsafeSetKeyType(keyType);
        d.unsafeSetValueType(valType);
        for (const auto& item : d) {
          Work kelem = {keyType, item.key()};
          Work velem = {valType, item.value()};
          to_process.emplace_back(std::move(kelem));
          to_process.emplace_back(std::move(velem));
        }
      } break;
      // in both cases the dynamic type is a class, and we are going to tag with
      // the dynamic type
      case InterfaceType::Kind:
      case ClassType::Kind: {
        auto obj = w.value.toObject();
        auto typ = obj->type(); // note: intentionally using the dynamic type,
                                // the static type is potentially less accurate
        for (size_t i = 0; i < typ->numAttributes(); ++i) {
          Work elem = {typ->getAttribute(i), obj->getSlot(i)};
          to_process.emplace_back(std::move(elem));
        }
      };
    }
  }
}

IValue lantern_read_pickle(
    const std::string& archive_name,
    caffe2::serialize::PyTorchStreamReader& stream_reader) {
  std::string picklename = archive_name + ".pkl";
  at::DataPtr pickle_ptr;
  size_t pickle_size = 0;
  std::tie(pickle_ptr, pickle_size) = stream_reader.getRecord(picklename);

  size_t bytes_read = 0;
  auto data = reinterpret_cast<const char*>(pickle_ptr.get());
  auto reader = [&](char* buffer, size_t len) -> size_t {
    if (bytes_read >= pickle_size) {
      return 0;
    }
    len = std::min(pickle_size - bytes_read, len);
    // Copy len bytes into buffer
    const char* start = data + bytes_read;
    std::memcpy(buffer, start, len);
    bytes_read += len;
    return len;
  };

  std::string tensor_dir_path = archive_name + "/";

  auto read_record = [&](const std::string& name) {
    std::string ss = tensor_dir_path + name;
    return std::get<0>(stream_reader.getRecord(ss));
  };

  LanternUnpickler unpickler(
      reader,
      nullptr,
      nullptr,
      std::move(read_record),
      c10::nullopt,
      false,
      LanternUnpickler::defaultTypeParser,
      std::move(nullptr));
  unpickler.set_version(stream_reader.version());
  return unpickler.parse_ivalue();
}


namespace {
template <typename T>
bool is(const Type& type) {
  if (type.kind() == T::Kind) {
    return true;
  }
  if (auto dyn = type.castRaw<c10::DynamicType>()) {
    return dyn->tag() == c10::DynamicTypeTrait<T>::tagValue();
  }
  return false;
}
} // namespace

static void restoreContainerTypeTags(
    const IValue& ivalue,
    const TypePtr& type) {
  if (is<DictType>(*type)) {
    auto dict = ivalue.toGenericDict();
    dict.unsafeSetKeyType(type->containedType(0));
    dict.unsafeSetValueType(type->containedType(1));
  } else if (is<ListType>(*type)) {
    ivalue.toList().unsafeSetElementType(type->containedType(0));
  } else {
    TORCH_CHECK(
        false, "Unknown type for tag restoration: " + type->annotation_str());
  }
}

void LanternUnpickler::readGlobal(
    const std::string& module_name,
    const std::string& class_name) {
  if (this->skip_next_read_global) {
    // See [NOTE] skip_next_read_global
    this->skip_next_read_global--;
    if (this->skip_next_read_global == 1) {
      // Pass through to the correct handler
    } else if (this->skip_next_read_global == 0) {
      // Corresponds to the type of `Tensor` being unpickled
      if (module_name != "torch" || class_name != "Tensor") {
        TORCH_WARN(
            "Trying to load a Subclassed Tensor, it will be converted to at::Tensor in C++");
      }
      stack_.emplace_back(int64_t(globals_.size() - 1));
      return;
    } else {
      TORCH_CHECK(false, "INVALID VALUES")
    }
  }
  // TODO [unpickler refactor] __main__ isn't used by the pickler anymore, this
  // is only here for bc-compatibility reasons
  if (module_name == "__main__") {
    if (class_name == "TensorID") {
      globals_.emplace_back([this] {
        auto setitem_data = stack_.back();
        stack_.pop_back();
        TORCH_INTERNAL_ASSERT(
            !tensor_table_.empty(),
            "Pickler tried to write a tensor but had no tensor table to write to");
        stack_.emplace_back(tensor_table_.at(setitem_data.toInt()));
      });
    } else if (class_name == "IntList") {
      globals_.emplace_back([this] {
        stack_.back().toList().unsafeSetElementType(IntType::get());
      });
    } else {
      TORCH_CHECK(false, "Unknown pickler class id", class_name);
    }
  } else if (module_name == "torch.jit._pickle") {
    if (class_name == "build_tensor_from_id") {
      globals_.emplace_back([this] {
        // Pop reduce arg off the stack
        auto data = stack_.back().toTupleRef().elements().at(0);
        stack_.pop_back();
        TORCH_CHECK(
            !tensor_table_.empty(),
            "Found a tensor table reference but Unpickler"
            " has no tensor table\n");
        stack_.emplace_back(tensor_table_.at(data.toInt()));
      });
    } else if (class_name == "restore_type_tag") {
      globals_.emplace_back([this] {
        auto tuple = stack_.back().toTuple();
        const auto& data = tuple->elements();
        auto type_str = data.at(1).toStringRef();
        stack_.pop_back();
        TypePtr type = nullptr;
        auto entry = type_cache_.find(type_str);
        if (entry != type_cache_.end()) {
          type = entry->second;
        } else {
          if (type_resolver_ == nullptr) {
            // If we haven't injected a custom way of retrieving types from
            // names, use a barebones type parser.
            type = type_parser_(type_str);
          } else {
            type = type_resolver_(type_str).type_;
          }
          type_cache_[type_str] = type;
        }
        // TODO: Use lookahead to avoid creating the tuple and immediately
        // destroying it here
        restoreContainerTypeTags(data.at(0), type);
        stack_.emplace_back(data.at(0));
      });
    } else {
      TypePtr elem_type = nullptr;
      if (class_name == "build_intlist") {
        elem_type = IntType::get();
      } else if (class_name == "build_tensorlist") {
        elem_type = TensorType::get();
      } else if (class_name == "build_doublelist") {
        elem_type = FloatType::get();
      } else if (class_name == "build_boollist") {
        elem_type = BoolType::get();
      } else {
        TORCH_CHECK(false, "Unknown pickler class id ", class_name);
      }
      // Unpickle a list specialization (e.g. List[Tensor], List[int], ...)
      globals_.emplace_back([this, elem_type] {
        // Pop reduce arg off the stack
        auto data = stack_.back().toTupleRef().elements().at(0).toList();
        stack_.pop_back();
        data.unsafeSetElementType(elem_type);
        stack_.emplace_back(std::move(data));
      });
    }
  } else if (
      module_name == "torch._utils" &&
      (class_name == "_rebuild_tensor_v2" ||
       class_name == "_rebuild_qtensor")) {
    // Unpickle a tensor
    bool quantized = class_name == "_rebuild_qtensor";
    rebuildTensor(quantized);
  } else if (
      module_name == "torch._tensor" &&
      (class_name == "_rebuild_from_type_v2")) {
    // Unpickle a Tensor with Python attributes or
    // a Subclassed Tensor.
    rebuildTensorFromTypeV2();
  } else if (
      module_name == "torch._utils" && class_name == "_rebuild_sparse_tensor") {
    rebuildSparseTensor();
  } else if (module_name == "builtins" && class_name == "complex") {
    globals_.emplace_back([this] {
      auto tuple = pop(stack_).toTuple();
      const auto& elems = tuple->elements();
      AT_ASSERT(elems.size() == 2);
      auto complex =
          c10::complex<double>(elems.at(0).toDouble(), elems.at(1).toDouble());
      stack_.emplace_back(complex);
    });

  } else if (module_name == "collections" && class_name == "OrderedDict") {
    // collections.OrderedDict is used in tensor serialization for a tensor's
    // backward hooks (but they are not actually saved with this Pickler)
    // Python's model.state_dict() is an OrderedDict, but this is not used
    // for model loading.
    globals_.emplace_back([this] {
      // The OrderedDict becomes a GenericDict. The inputs which are in
      // stack.back() are fully ignored, but they are empty anyways.
      stack_.back() = c10::impl::GenericDict(AnyType::get(), AnyType::get());
    });
  } else if (module_name == "torch" && class_name == "device") {
    globals_.emplace_back([this] {
      auto device_string = stack_.back().toTupleRef().elements().at(0);
      stack_.pop_back();
      stack_.emplace_back(c10::Device(device_string.toStringRef()));
    });
    stack_.emplace_back(int64_t(globals_.size() - 1));
    return;
  } else if (module_name == "torch.distributed.rpc" && class_name == "rref") {
#ifdef USE_RPC
    return rebuildRRef();
#else
    TORCH_INTERNAL_ASSERT(
        false,
        "RRef unpickling is only supported with the distributed package");
#endif
  } else if (module_name == "torch") {
    // Try to manually resolve several global enums
    // NOTE: this does not put a global into the global table,
    // like the other branches here because no REDUCE or BUILD will
    // be called on this value. Instead, we just put it on the stack
    // and return early
    std::optional<c10::ScalarType> scalar_type;
#define CHECK_SCALAR(_, name)          \
  if (class_name == #name "Storage") { \
    scalar_type = c10::k##name;        \
  }
    AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(CHECK_SCALAR)
#undef CHECK_SCALAR
    if (scalar_type.has_value()) {
      stack_.emplace_back(int64_t(*scalar_type));
      return;
    }

    std::optional<at::QScheme> qscheme;
    for (int i = 0; i < at::COMPILE_TIME_NUM_QSCHEMES; ++i) {
      if (class_name == toString(static_cast<at::QScheme>(i))) {
        qscheme = static_cast<at::QScheme>(i);
      }
    }
    if (qscheme.has_value()) {
      stack_.emplace_back(int64_t(*qscheme));
      return;
    }
    TORCH_CHECK(
        false,
        "Unpickler found unknown torch global, 'torch.",
        class_name,
        "'");
  } else {
    TORCH_CHECK(
        type_resolver_,
        "Unpickler found unknown type ",
        module_name,
        ".",
        class_name);
    at::StrongTypePtr type =
        type_resolver_(c10::QualifiedName(module_name, class_name));
    if (auto enum_type = type.type_->cast<c10::EnumType>()) {
      globals_.emplace_back([this, enum_type] {
        auto val = stack_.back();
        stack_.pop_back();
        for (const auto& p : enum_type->enumNamesValues()) {
          if (p.second == val) {
            auto enum_holder = c10::make_intrusive<at::ivalue::EnumHolder>(
                enum_type, p.first, p.second);
            stack_.emplace_back(std::move(enum_holder));
            return;
          }
        }
      });
    } else {
      // Otherwise, global is a class/object type.
      globals_.emplace_back([this, type] {
        auto val = stack_.back();
        stack_.pop_back();
        auto obj = obj_loader_(type, val);
        stack_.emplace_back(std::move(obj));
      });
    }
  }
  stack_.emplace_back(int64_t(globals_.size() - 1));
}

PickleOpCode LanternUnpickler::readInstruction() {
  auto opcode = readOpCode();
  switch (opcode) {
    case PickleOpCode::EMPTY_LIST: {
      stack_.emplace_back(c10::impl::GenericList(AnyType::get()));
    } break;
    case PickleOpCode::EMPTY_TUPLE: {
      if (empty_tuple_.isNone()) {
        // we only need one object, since tuples are not mutable.
        empty_tuple_ = c10::ivalue::Tuple::create(std::vector<IValue>());
      }
      stack_.emplace_back(empty_tuple_);
    } break;
    case PickleOpCode::BINPUT: {
      size_t memo_id = read<uint8_t>();
      setInput(memo_id);
    } break;
    case PickleOpCode::LONG_BINPUT: {
      TORCH_CHECK(
          std::numeric_limits<size_t>::max() >=
              std::numeric_limits<uint32_t>::max(),
          "Found a LONG_BINPUT opcode, but size_t on this system is "
          "not big enough to decode it");
      size_t memo_id = read<uint32_t>();
      setInput(memo_id);
    } break;
    case PickleOpCode::MARK: {
      // Mark location of the container ivalue in the stack
      marks_.push_back(stack_.size());
    } break;
    case PickleOpCode::NEWTRUE: {
      stack_.emplace_back(true);
    } break;
    case PickleOpCode::NEWFALSE: {
      stack_.emplace_back(false);
    } break;
    case PickleOpCode::NONE: {
      stack_.emplace_back();
    } break;
    case PickleOpCode::BININT1: {
      uint8_t value = read<uint8_t>();
      stack_.emplace_back(int64_t(value));
    } break;
    case PickleOpCode::BININT2: {
      uint16_t value = from_le16(read<uint16_t>());
      stack_.emplace_back(int64_t(value));
    } break;
    case PickleOpCode::BININT: {
      int32_t value = from_le32(read<int32_t>());
      stack_.emplace_back(int64_t(value));
    } break;
    case PickleOpCode::LONG1: {
      // Only read LONG1s with 8 as the length
      uint8_t length = read<uint8_t>();
      TORCH_CHECK(length == 8, "Expected length to be 8, got ", int(length));
      stack_.emplace_back(int64_t(from_le64(read<int64_t>())));
    } break;
    case PickleOpCode::BINUNICODE: {
      uint32_t length = from_le32(read<uint32_t>());
      stack_.emplace_back(readBytes(length));
    } break;
    case PickleOpCode::BINUNICODE8: {
      int64_t length = from_le64(read<int64_t>());
      stack_.emplace_back(readBytes(length));
    } break;
    case PickleOpCode::BINFLOAT:
      stack_.emplace_back(readFloat());
      break;
    case PickleOpCode::TUPLE: {
      TORCH_CHECK(!marks_.empty(), "Parsing error: marks_ is empty");
      size_t start = marks_.back();
      marks_.pop_back();
      std::vector<IValue> elements;
      TORCH_CHECK(
          stack_.size() >= start,
          "Parsing error: wrong start index ",
          start,
          " for stack_ of size ",
          stack_.size());
      const auto tupleSize = stack_.size() - start;
      switch (tupleSize) {
        case 3: {
          auto e3 = pop(stack_);
          auto e2 = pop(stack_);
          auto e1 = pop(stack_);
          stack_.emplace_back(c10::ivalue::Tuple::create(
              std::move(e1), std::move(e2), std::move(e3)));
          break;
        }
        case 2: {
          auto e2 = pop(stack_);
          auto e1 = pop(stack_);
          stack_.emplace_back(
              c10::ivalue::Tuple::create(std::move(e1), std::move(e2)));
          break;
        }
        case 1:
          stack_.emplace_back(c10::ivalue::Tuple::create(pop(stack_)));
          break;
        default: {
          elements.reserve(stack_.size() - start);
          auto start_it = stack_.begin() + static_cast<std::ptrdiff_t>(start);
          for (auto it = start_it; it != stack_.end(); ++it) {
            elements.emplace_back(std::move(*it));
          }
          stack_.erase(start_it, stack_.end());
          stack_.emplace_back(c10::ivalue::Tuple::create(std::move(elements)));
          break;
        }
      }
    } break;
    case PickleOpCode::TUPLE1: {
      TORCH_CHECK(
          !stack_.empty(),
          "Parsing error: stack_ contains ",
          stack_.size(),
          " elements, at least 1 expected");
      stack_.emplace_back(c10::ivalue::Tuple::create(pop(stack_)));
    } break;
    case PickleOpCode::TUPLE2: {
      TORCH_CHECK(
          stack_.size() > 1,
          "Parsing error: stack_ contains ",
          stack_.size(),
          " elements, at least 2 expected");
      auto e2 = pop(stack_);
      auto e1 = pop(stack_);
      stack_.emplace_back(
          c10::ivalue::Tuple::create(std::move(e1), std::move(e2)));
    } break;
    case PickleOpCode::TUPLE3: {
      TORCH_CHECK(
          stack_.size() > 2,
          "Parsing error: stack_ contains ",
          stack_.size(),
          " elements, at least 3 expected");
      auto e3 = pop(stack_);
      auto e2 = pop(stack_);
      auto e1 = pop(stack_);
      stack_.emplace_back(c10::ivalue::Tuple::create(
          std::move(e1), std::move(e2), std::move(e3)));
    } break;
    case PickleOpCode::EMPTY_DICT:
      stack_.emplace_back(
          c10::impl::GenericDict(AnyType::get(), AnyType::get()));
      break;
    case PickleOpCode::APPENDS: {
      TORCH_CHECK(!marks_.empty(), "Parsing error: marks_ is empty");
      size_t start = marks_.back();
      TORCH_CHECK(
          start > 0 && start <= stack_.size(),
          "Parsing error: wrong start index ",
          start,
          " for stack_ of size ",
          stack_.size());
      auto list_ivalue = stack_.at(start - 1);
      readList(list_ivalue);
    } break;
    case PickleOpCode::APPEND: {
      TORCH_CHECK(
          stack_.size() >= 2, "Parsing error: missing elements in stack_.");
      auto list_ivalue = stack_.at(stack_.size() - 2);
      readListElements(list_ivalue, stack_.size() - 1);
    } break;
    case PickleOpCode::LIST: {
      IValue list_ivalue = c10::impl::GenericList(AnyType::get());
      readList(list_ivalue);
      stack_.push_back(std::move(list_ivalue));
    } break;
    case PickleOpCode::DICT: {
      TORCH_CHECK(!marks_.empty(), "Parsing error: marks_ is empty");
      size_t start = marks_.back();
      marks_.pop_back();
      TORCH_CHECK(
          stack_.size() > start,
          "Parsing error: wrong start index ",
          start,
          " for stack_ which of size ",
          stack_.size());
      auto dict = c10::impl::GenericDict(AnyType::get(), AnyType::get());
      TORCH_CHECK(
          (stack_.size() - start) % 2 == 0,
          "Parsing error: stack_ is of size ",
          stack_.size(),
          " and start index is ",
          start,
          ", but stack_ is iterated by two elements at a time");
      for (size_t i = start; i < stack_.size(); i += 2) {
        dict.insert_or_assign(stack_[i], stack_[i + 1]);
      }
      stack_.erase(
          stack_.begin() + static_cast<std::ptrdiff_t>(start), stack_.end());
      stack_.emplace_back(std::move(dict));
    } break;
    case PickleOpCode::SETITEMS: {
      TORCH_CHECK(!marks_.empty(), "Parsing error: marks_ is empty");
      size_t start = marks_.back();
      marks_.pop_back();
      TORCH_CHECK(
          start > 0 && start <= stack_.size(),
          "Parsing error: wrong start index for stack_");
      auto dict = stack_.at(start - 1).toGenericDict();
      TORCH_CHECK(
          (stack_.size() - start) % 2 == 0,
          "Parsing error: stack_ is of size ",
          stack_.size(),
          " and start index is ",
          start,
          ", but stack_ is iterated by two elemenst at a time");
      for (size_t i = start; i < stack_.size(); i += 2) {
        dict.insert_or_assign(stack_[i], stack_[i + 1]);
      }
      stack_.erase(
          stack_.begin() + static_cast<std::ptrdiff_t>(start), stack_.end());
    } break;
    case PickleOpCode::BINGET: {
      auto pos = read<uint8_t>();
      TORCH_CHECK(
          memo_table_.size() > pos,
          "Parsing error: out of bounds access at ",
          (size_t)pos,
          " to memo_table_ which is of size ",
          memo_table_.size());
      stack_.push_back(memo_table_.at(pos));
    } break;
    case PickleOpCode::LONG_BINGET: {
      auto pos = read<uint32_t>();
      TORCH_CHECK(
          memo_table_.size() > pos,
          "Parsing error: out of bounds access at ",
          (size_t)pos,
          " to memo_table_ which is of size ",
          memo_table_.size());
      stack_.push_back(memo_table_.at(pos));
    } break;
    case PickleOpCode::STOP:
      break;
    case PickleOpCode::GLOBAL: {
      // Module name, it's not needed for anything
      auto module_name = readString();
      auto class_name = readString();
      readGlobal(module_name, class_name);
    } break;
    case PickleOpCode::NEWOBJ: {
      TORCH_CHECK(!stack_.empty(), "Parsing error: stack_ is empty");
      // pop empty tuple, the actual action is stored in the globals_stack_
      stack_.pop_back();
    } break;
    // because we have NEWOBJ do nothing, BUILD and REDUCE end up doing
    // the same thing
    case PickleOpCode::BUILD:
    case PickleOpCode::REDUCE: {
      // stack is: <functor_idx> <functor_arg>
      // extract <functor_idx> and remove from the stack:
      TORCH_CHECK(
          stack_.size() > 1,
          "Parsing error: stack_ contains ",
          stack_.size(),
          " elements, at least 2 expected");

      // In the OrderedDict case, the id has already been materialized
      // and added to the stack, thus there's no <functor_idx> but a Dict
      // there, in this case we can just pop the functor args and break.
      // The functor args in this case contain some other metadata like
      // '{_metadata: {: {version: 1}}}' which seem to be safe to ignore.
      if (stack_.at(stack_.size() - 2).isGenericDict()) {
        stack_.pop_back();
        break;
      }
    
      std::swap(*(stack_.end() - 2), *(stack_.end() - 1));
      size_t idx = stack_.back().toInt();
      stack_.pop_back();
      // stack is: <functor_arg>
      TORCH_CHECK(
          idx < globals_.size(),
          "Parsing error: out of bounds access to globals_");
      globals_.at(idx)();
    } break;
    case PickleOpCode::BINPERSID: {
      TORCH_CHECK(!stack_.empty(), "Parsing error: stack_ is empty");
      auto tuple = pop(stack_).toTuple();
      const auto& args = tuple->elements();
      AT_ASSERT(
          args.at(0).toStringRef() == "storage",
          "unknown PERSID key ",
          args.at(0).toStringRef());
      at::ScalarType type = args.at(1).toScalarType();
      const std::string& key = args.at(2).toStringRef();

      at::Device device(args.at(3).toStringRef());
      // remap device location if it's not meta
      if (device_ && !device.is_meta()) {
        device = *device_;
      }

      at::Storage storage;
      if (storage_context_ != nullptr && storage_context_->hasStorage(key)) {
        // for torch.package logic where storage may be loaded already
        storage = storage_context_->getStorage(key);
      } else {
        int64_t numel = args.at(4).toInt();
        auto dtype = scalarTypeToTypeMeta(type);

        at::DataPtr storage_ptr;
        if (numel > 0) {
          // If there are no elements in the tensor, there's no point in
          // reading a zero (0) byte file from the input stream and paying
          // that cost.
          storage_ptr = read_record_(key);
        }

        storage = at::Storage(
            c10::Storage::use_byte_size_t(),
            numel * dtype.itemsize(),
            std::move(storage_ptr),
            /*allocator=*/nullptr,
            /*resizable=*/false); // NB: we didn't set any allocator for the
                                  // tensor
        if (storage_context_ != nullptr) {
          storage_context_->addStorage(key, storage);
        }
      }

      auto options = at::device(at::kCPU).dtype(type);
      if (use_storage_device_) {
        options = options.device(storage.device());
        device = storage.device();
      }

      at::Tensor tensor;
      if (options.backend() == c10::Backend::QuantizedCPU) {
        tensor = at::_empty_affine_quantized({}, options, 0, 0)
                     .set_(storage, 0, {}, {});
      } else {
        tensor = at::empty({0}, options).set_(storage);
      }

      if (device.is_cuda() || device.is_xpu() || device.is_meta() ||
          device.is_hpu() || device.is_mps() || device.is_privateuseone()) {
        tensor = tensor.to(device, tensor.scalar_type());
      } else if (device.type() != DeviceType::CPU) {
        TORCH_CHECK(
            false,
            "supported devices include CPU, CUDA, HPU and ",
            c10::get_privateuse1_backend(),
            " however got ",
            DeviceTypeName(device.type(), false));
      }
      stack_.emplace_back(std::move(tensor));
    } break;
    case PickleOpCode::SETITEM: {
      // At this OpCode, stack looks like
      // | Stack Bottom |
      // | ......       |
      // | Dict         | -> (stack_size - 3)
      // | Key          | -> (stack_size - 2)
      // | Value        | -> (stack_size - 1)
      TORCH_CHECK(
          stack_.size() >= 3,
          "Parsing error: stack doesn't have enough elements");

      auto stack_size = stack_.size();
      auto dict_pos = stack_size - 3;
      auto key_pos = stack_size - 2;
      auto val_pos = stack_size - 1;

      TORCH_CHECK(
          (dict_pos < stack_size) && (key_pos < stack_size) &&
              (val_pos < stack_size),
          "Parsing error: attempted out-of-bounds access while processing SETITEM opcode");

      auto dict = stack_.at(dict_pos).toGenericDict();
      dict.insert_or_assign(stack_.at(key_pos), stack_.at(val_pos));
      stack_.erase(
          stack_.begin() + static_cast<std::ptrdiff_t>(key_pos), stack_.end());
    } break;
    default: {
      TORCH_CHECK(
          false,
          "Unknown opcode for unpickling at ",
          // NOLINTNEXTLINE(performance-no-int-to-ptr)
          reinterpret_cast<void*>(opcode),
          ": ",
          int(static_cast<uint8_t>(opcode)));
    } break;
  }
  return opcode;
}

void LanternUnpickler::run() {
  // Expect a PROTO opcode and protocol number at the start of blob
  auto opcode = readOpCode();
  TORCH_CHECK(
      opcode == PickleOpCode::PROTO,
      "Expected PROTO opcode at the start"
      " of pickle archive, found ",
      int(static_cast<uint8_t>(opcode)));
  uint8_t protocol = read<uint8_t>();
  TORCH_CHECK(
      protocol == 2,
      "Only Pickle protocol 2 is supported, found protocol = ",
      protocol);

  while (true) {
    PickleOpCode opcode = readInstruction();
    if (opcode == PickleOpCode::STOP) {
      return;
    }
  }
}

IValue LanternUnpickler::parse_ivalue() {
  run();
  TORCH_CHECK(
      stack_.size() == 1,
      "Unpickler expected 1 element on the stack, but found ",
      stack_.size());
  if (version_ <= 2) {
    // See [type tag serialization]
    restoreAccurateTypeTagsIfPossible(stack_[0]);
  }
  return stack_[0];
}

} // namespace jit
} // namespace torch