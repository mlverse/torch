#include <ATen/ATen.h>
#include <ATen/core/Dict.h>
#ifdef USE_RPC
#include <torch/csrc/distributed/rpc/rref_context.h>
#endif
#include <torch/csrc/jit/api/function_impl.h>
#include <torch/csrc/jit/mobile/type_parser.h>
#include <torch/csrc/jit/serialization/pickler.h>
#include <torch/csrc/jit/serialization/storage_context.h>
#include <torch/csrc/jit/serialization/unpickler.h>
#include <string>

#ifdef _WIN32
#define LANTERN_API __declspec(dllexport)
#else
#define LANTERN_API
#endif

namespace torch {
namespace jit {

// See Python's pickletools.py for a detailed description of each of these codes
enum class PickleOpCode2 : char {
  MARK = '(',
  STOP = '.',
  POP = '0',
  POP_MARK = '1',
  DUP = '2',
  FLOAT = 'F',
  INT = 'I',
  BININT = 'J',
  BININT1 = 'K',
  LONG = 'L',
  BININT2 = 'M',
  NONE = 'N',
  PERSID = 'P',
  BINPERSID = 'Q',
  REDUCE = 'R',
  STRING = 'S',
  BINSTRING = 'T',
  SHORT_BINSTRING = 'U',
  // NB: Avoid using UNICODE as it is a macro in the Windows API
  UNICODE_ = 'V',
  BINUNICODE = 'X',
  APPEND = 'a',
  BUILD = 'b',
  GLOBAL = 'c',
  DICT = 'd',
  EMPTY_DICT = '}',
  APPENDS = 'e',
  GET = 'g',
  BINGET = 'h',
  INST = 'i',
  LONG_BINGET = 'j',
  LIST = 'l',
  EMPTY_LIST = ']',
  OBJ = 'o',
  PUT = 'p',
  BINPUT = 'q',
  LONG_BINPUT = 'r',
  SETITEM = 's',
  TUPLE = 't',
  EMPTY_TUPLE = ')',
  SETITEMS = 'u',
  BINFLOAT = 'G',

  // Protocol 2
  PROTO = char('\x80'),
  NEWOBJ = '\x81',
  EXT1 = '\x82',
  EXT2 = '\x83',
  EXT4 = '\x84',
  TUPLE1 = '\x85',
  TUPLE2 = '\x86',
  TUPLE3 = '\x87',
  NEWTRUE = '\x88',
  NEWFALSE = '\x89',
  LONG1 = '\x8a',
  LONG4 = '\x8b',

  // Protocol 3 (Python 3.x)
  BINBYTES = 'B',
  SHORT_BINBYTES = 'C',

  // Protocol 4
  SHORT_BINUNICODE = char('\x8c'),
  BINUNICODE8 = '\x8d',
  BINBYTES8 = '\x8e',
  EMPTY_SET = '\x8f',
  ADDITEMS = '\x90',
  FROZENSET = '\x91',
  NEWOBJ_EX = '\x92',
  STACK_GLOBAL = '\x93',
  MEMOIZE = '\x94',
  FRAME = '\x95'
};

using TypeResolver =
    std::function<c10::StrongTypePtr(const c10::QualifiedName&)>;

using ObjLoader = std::function<
    c10::intrusive_ptr<c10::ivalue::Object>(at::StrongTypePtr, IValue)>;

class DeserializationStorageContext;

// [unpickler refactor] there is some cruft around PickleOpCode2::BUILD,
// PickleOpCode2::NEWOBJ, and the last_opcode_ member below that should be
// deleted at some point, the Pickler doesn't produce it and it's only around to
// support models saved before 1.1
class LANTERN_API LanternUnpickler {
  TH_DISALLOW_COPY_AND_ASSIGN(LanternUnpickler);

  using TypeParserT = c10::TypePtr (*)(const std::string&);

 public:
  // tensors inside the pickle are references to the tensor_table.
  // class_resolver is to resolve strong class type, type_resolver_ is
  // to resolve any JIT type. class_resolver and type_resolver are not merged
  // here because some use cases need to get strong class type that
  // type_resolver_ can not return.
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  LanternUnpickler(
      std::function<size_t(char*, size_t)> reader,
      TypeResolver type_resolver,
      c10::ArrayRef<at::Tensor> tensor_table,
      TypeParserT type_parser = defaultTypeParser)
      : reader_(std::move(reader)),
        tensor_table_(tensor_table),
        type_resolver_(std::move(type_resolver)),
        use_storage_device_(false),
        type_parser_(type_parser),
        version_(caffe2::serialize::kProducedFileFormatVersion) {}

  // tensors inside the pickle contain meta-data, the raw tensor
  // dead is retrieved by calling `read_record`.
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  LanternUnpickler(
      std::function<size_t(char*, size_t)> reader,
      TypeResolver type_resolver,
      ObjLoader obj_loader,
      std::function<at::DataPtr(const std::string&)> read_record,
      c10::optional<at::Device> device,
      bool use_storage_device = false,
      TypeParserT type_parser = defaultTypeParser,
      std::shared_ptr<DeserializationStorageContext> storage_context = nullptr)
      : reader_(std::move(reader)),
        tensor_table_(),
        type_resolver_(std::move(type_resolver)),
        obj_loader_(std::move(obj_loader)),
        read_record_(std::move(read_record)),
        // NOLINTNEXTLINE(performance-move-const-arg)
        device_(std::move(device)),
        use_storage_device_(use_storage_device),
        type_parser_(type_parser),
        storage_context_(std::move(storage_context)),
        version_(caffe2::serialize::kProducedFileFormatVersion) {}

  // consume the pickle stream, producing an IValue from the contents.
  // Type Tags: the pickler will restore the type tags on
  // List and Dict objects when possible IValue is an Object.
  // Otherwise, Dict and List objects will end up with Any as their tag.
  // If you know the type of the ivalue, tags can be restored with
  // restoreAccurateTypeTags
  IValue parse_ivalue();

  // [type tag serialization]
  // This is used to determine whether to restore type tags be recursively
  // descending into the returned stack object (if version_number <= 2), or
  // if version_number >= 3, to use the type strings included in the pickle
  // archive for container types. By default this is set to
  // `kProducedFileFormatVersion` so unless you're loading a pickle file
  // from alongside a corresponding `version` file, you don't need to set
  // the version manually.
  void set_version(uint64_t version_number) {
    version_ = version_number;
  }

  static c10::TypePtr defaultTypeParser(const std::string& str) {
    ScriptTypeParser parser;
    return parser.parseType(str);
  }

 private:
  // No arguments ensures that a template argument must be specified
  // so that the number of bytes read / type read is explicit
  template <typename T>
  T read() {
    T item;
    if (sizeof(T) <= buffer_remaining_) {
      // Fast path: entirely from buffer.
      memcpy(&item, buffer_.data() + buffer_pos_, sizeof(T));
      buffer_remaining_ -= sizeof(T);
      buffer_pos_ += sizeof(T);
    } else {
      // Don't over-template the slow path, to avoid code size bloat.
      readSlowWithBuffer(reinterpret_cast<char*>(&item), sizeof(T));
    }
    return item;
  }
  void readSlowWithBuffer(char* dest, size_t sz);
  std::string readBytes(size_t num_bytes);

  double readFloat();
  void readGlobal(
      const std::string& module_name,
      const std::string& class_name);
  void rebuildTensor(bool quantized);
  void rebuildSparseTensor();
#ifdef USE_DISTRIBUTED
  void rebuildRRef();
#endif
  PickleOpCode2 readInstruction();
  PickleOpCode2 readOpCode() {
    return static_cast<PickleOpCode2>(read<uint8_t>());
  }
  std::string readString();
  void readList(IValue list_ivalue);
  void setInput(size_t memo_id);
  void run();

  // Returns the number of bytes read. This should statefully
  // remember the position. Don't call reader_ directly.
  std::function<size_t(char*, size_t)> reader_;
  // Small buffer to avoid calling reader_ on a per-byte basis.
  std::array<char, 256> buffer_;
  size_t buffer_pos_{0};
  size_t buffer_remaining_{0};

  std::vector<IValue> stack_;

  // globals are represented on the stack as IValue integer indices
  // into this list
  std::vector<std::function<void(void)>> globals_;
  std::vector<IValue> memo_table_;
  std::vector<size_t> marks_;
  c10::ArrayRef<at::Tensor> tensor_table_;

  // When deserializing types on lists and dicts, cache the type here
  // so we don't have to parse the same type multiple times. Strings
  // are already de-duplicated and replaced with BINGETs in the
  // pickler, so we can just use the actual data pointer of each string.
  std::unordered_map<std::string, c10::TypePtr> type_cache_;

  // optionally nullptr, needs to be present for creating classes
  TypeResolver type_resolver_;
  ObjLoader obj_loader_;
  IValue empty_tuple_;

  std::function<at::DataPtr(const std::string&)> read_record_;
  c10::optional<at::Device> device_;
  // When set to true, Unpickler will ignore the pickled device and use the
  // device of the DataPtr returned by the read_record_ function. The default
  // value of this flag is false.
  const bool use_storage_device_;

  TypeParserT type_parser_{defaultTypeParser};

  // Used for torch.package to enable sharing of storages across
  // ScriptModules and eager modules
  std::shared_ptr<DeserializationStorageContext> storage_context_;

  // See [type tag serialization]
  uint64_t version_;
};

void restoreAccurateTypeTags(const IValue& root, const c10::TypePtr& type_tag);

} // namespace jit
} // namespace torch


namespace torch {
namespace jit {

using ::c10::IValue;

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
        TORCH_CHECK(!w.value.toSymInt().is_symbolic());
        // no op, there is nothing to tag
        break;
      case c10::SymFloatType::Kind:
        TORCH_CHECK(!w.value.toSymFloat().is_symbolic());
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
        for (const IValue item : lst) {
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

void restoreContainerTypeTags(const IValue& ivalue, const TypePtr& type) {
  if (is<DictType>(*type)) {
    auto dict = ivalue.toGenericDict();
    dict.unsafeSetKeyType(type->containedType(0));
    dict.unsafeSetValueType(type->containedType(1));
  } else if (is<ListType>(*type)) {
    ivalue.toList().unsafeSetElementType(type->containedType(0));
  } else {
    AT_ERROR("Unknown type for tag restoration: " + type->annotation_str());
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

double LanternUnpickler::readFloat() {
  AT_ASSERT(sizeof(double) == 8);
  double big_endian = read<double>();
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  double little_endian;

  // Pickle floats are big endian, so reverse the bytes
  auto big_endian_ptr = reinterpret_cast<const char*>(&big_endian);
  std::reverse_copy(
      big_endian_ptr,
      big_endian_ptr + sizeof(big_endian),
      reinterpret_cast<char*>(&little_endian));

  return little_endian;
}

void LanternUnpickler::run() {
  // Expect a PROTO opcode and protocol number at the start of blob
  auto opcode = readOpCode();
  TORCH_CHECK(
      opcode == PickleOpCode2::PROTO,
      "Expected PROTO opcode at the start"
      " of pickle archive, found ",
      int(static_cast<uint8_t>(opcode)));
  uint8_t protocol = read<uint8_t>();
  TORCH_CHECK(
      protocol == 2,
      "Only Pickle protocol 2 is supported, found protocol = ",
      protocol);

  while (true) {
    PickleOpCode2 opcode = readInstruction();
    if (opcode == PickleOpCode2::STOP) {
      return;
    }
  }
}
void LanternUnpickler::setInput(size_t memo_id) {
  AT_ASSERT(!stack_.empty());
  if (memo_id >= memo_table_.size()) {
    memo_table_.insert(
        memo_table_.end(), memo_id - memo_table_.size(), IValue());
    memo_table_.push_back(stack_.back());
  } else {
    memo_table_[memo_id] = stack_.back();
  }
}

// emplace_back on bool vectors does not exist on some systems
// avoid it by calling push_back for bool
template <typename T>
inline void append(std::vector<T>& a, T&& e) {
  a.emplace_back(std::forward<T>(e));
}
template <>
inline void append<bool>(std::vector<bool>& a, bool&& e) {
  a.push_back(e);
}

static std::vector<int64_t> tupleToIntList(const IValue& v) {
  return fmap(v.toTupleRef().elements(), [](const IValue& v) -> int64_t {
    return v.toInt();
  });
}

// note we cannot use toIntList, toDoubleList because during unpickling the
// lists are not yet tagged
template <typename T>
static std::vector<T> convertList(const IValue& v) {
  return fmap(v.toListRef(), [](const IValue& elem) { return elem.to<T>(); });
}

PickleOpCode2 LanternUnpickler::readInstruction() {
  auto opcode = readOpCode();
  switch (opcode) {
    case PickleOpCode2::EMPTY_LIST: {
      stack_.emplace_back(c10::impl::GenericList(AnyType::get()));
    } break;
    case PickleOpCode2::EMPTY_TUPLE: {
      if (empty_tuple_.isNone()) {
        // we only need one object, since tuples are not mutable.
        empty_tuple_ = c10::ivalue::Tuple::create(std::vector<IValue>());
      }
      stack_.emplace_back(empty_tuple_);
    } break;
    case PickleOpCode2::BINPUT: {
      size_t memo_id = read<uint8_t>();
      setInput(memo_id);
    } break;
    case PickleOpCode2::LONG_BINPUT: {
      TORCH_CHECK(
          std::numeric_limits<size_t>::max() >=
              std::numeric_limits<uint32_t>::max(),
          "Found a LONG_BINPUT opcode, but size_t on this system is "
          "not big enough to decode it");
      size_t memo_id = read<uint32_t>();
      setInput(memo_id);
    } break;
    case PickleOpCode2::MARK: {
      // Mark location of the container ivalue in the stack
      marks_.push_back(stack_.size());
    } break;
    case PickleOpCode2::NEWTRUE: {
      stack_.emplace_back(true);
    } break;
    case PickleOpCode2::NEWFALSE: {
      stack_.emplace_back(false);
    } break;
    case PickleOpCode2::NONE: {
      stack_.emplace_back(IValue());
    } break;
    case PickleOpCode2::BININT1: {
      uint8_t value = read<uint8_t>();
      stack_.emplace_back(int64_t(value));
    } break;
    case PickleOpCode2::BININT2: {
      uint16_t value = read<uint16_t>();
      stack_.emplace_back(int64_t(value));
    } break;
    case PickleOpCode2::BININT: {
      int32_t value = read<int32_t>();
      stack_.emplace_back(int64_t(value));
    } break;
    case PickleOpCode2::LONG1: {
      // Only read LONG1s with 8 as the length
      uint8_t length = read<uint8_t>();
      TORCH_CHECK(length == 8, "Expected length to be 8, got ", int(length));
      stack_.emplace_back(int64_t(read<int64_t>()));
    } break;
    case PickleOpCode2::BINUNICODE: {
      uint32_t length = read<uint32_t>();
      stack_.emplace_back(readBytes(length));
    } break;
    case PickleOpCode2::BINFLOAT:
      stack_.emplace_back(readFloat());
      break;
    case PickleOpCode2::TUPLE: {
      TORCH_CHECK(!marks_.empty(), "Parsing error: marks_ is empty");
      size_t start = marks_.back();
      marks_.pop_back();
      std::vector<IValue> elements;
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
          auto start_it = stack_.begin() + start;
          for (auto it = start_it; it != stack_.end(); ++it) {
            elements.emplace_back(std::move(*it));
          }
          stack_.erase(start_it, stack_.end());
          stack_.emplace_back(c10::ivalue::Tuple::create(std::move(elements)));
          break;
        }
      }
    } break;
    case PickleOpCode2::TUPLE1: {
      stack_.emplace_back(c10::ivalue::Tuple::create(pop(stack_)));
    } break;
    case PickleOpCode2::TUPLE2: {
      auto e2 = pop(stack_);
      auto e1 = pop(stack_);
      stack_.emplace_back(
          c10::ivalue::Tuple::create(std::move(e1), std::move(e2)));
    } break;
    case PickleOpCode2::TUPLE3: {
      auto e3 = pop(stack_);
      auto e2 = pop(stack_);
      auto e1 = pop(stack_);
      stack_.emplace_back(c10::ivalue::Tuple::create(
          std::move(e1), std::move(e2), std::move(e3)));
    } break;
    case PickleOpCode2::EMPTY_DICT:
      stack_.emplace_back(
          c10::impl::GenericDict(AnyType::get(), AnyType::get()));
      break;
    case PickleOpCode2::APPENDS: {
      TORCH_CHECK(!marks_.empty(), "Parsing error: marks_ is empty");
      size_t start = marks_.back();
      TORCH_CHECK(
          start > 0 && start <= stack_.size(),
          "Parsing error: wrong start index for stack_");
      auto list_ivalue = stack_.at(start - 1);
      readList(list_ivalue);
    } break;
    case PickleOpCode2::LIST: {
      IValue list_ivalue = c10::impl::GenericList(AnyType::get());
      readList(list_ivalue);
      stack_.push_back(std::move(list_ivalue));
    } break;
    case PickleOpCode2::DICT: {
      TORCH_CHECK(!marks_.empty(), "Parsing error: marks_ is empty");
      size_t start = marks_.back();
      marks_.pop_back();
      auto dict = c10::impl::GenericDict(AnyType::get(), AnyType::get());
      for (size_t i = start; i < stack_.size(); i += 2) {
        dict.insert_or_assign(stack_[i], stack_[i + 1]);
      }
      stack_.erase(stack_.begin() + start, stack_.end());
      stack_.emplace_back(std::move(dict));
    } break;
    case PickleOpCode2::SETITEMS: {
      TORCH_CHECK(!marks_.empty(), "Parsing error: marks_ is empty");
      size_t start = marks_.back();
      marks_.pop_back();
      TORCH_CHECK(
          start > 0 && start <= stack_.size(),
          "Parsing error: wrong start index for stack_");
      auto dict = stack_.at(start - 1).toGenericDict();
      for (size_t i = start; i < stack_.size(); i += 2) {
        dict.insert_or_assign(stack_[i], stack_[i + 1]);
      }
      stack_.erase(stack_.begin() + start, stack_.end());
    } break;
    case PickleOpCode2::BINGET: {
      stack_.push_back(memo_table_.at(read<uint8_t>()));
    } break;
    case PickleOpCode2::LONG_BINGET: {
      stack_.push_back(memo_table_.at(read<uint32_t>()));
    } break;
    case PickleOpCode2::STOP:
      break;
    case PickleOpCode2::GLOBAL: {
      // Module name, it's not needed for anything
      auto module_name = readString();
      auto class_name = readString();
      readGlobal(module_name, class_name);
    } break;
    case PickleOpCode2::NEWOBJ: {
      // pop empty tuple, the actual action is stored in the globals_stack_
      stack_.pop_back();
    } break;
    // because we have NEWOBJ do nothing, BUILD and REDUCE end up doing
    // the same thing
    case PickleOpCode2::BUILD:
    case PickleOpCode2::REDUCE: {
      // stack is: <functor_idx> <functor_arg>
      // extract <functor_idx> and remove from the stack:
      
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
    case PickleOpCode2::BINPERSID: {
      auto tuple = pop(stack_).toTuple();
      const auto& args = tuple->elements();
      AT_ASSERT(
          args.at(0).toStringRef() == "storage",
          "unknown PERSID key ",
          args.at(0).toStringRef());
      at::ScalarType type = args.at(1).toScalarType();
      const std::string& key = args.at(2).toStringRef();

      at::Device device(args.at(3).toStringRef());
      if (device_) {
        device = *device_;
      }

      at::Storage storage;
      if (storage_context_ != nullptr && storage_context_->hasStorage(key)) {
        // for torch.package logic where storage may be loaded already
        storage = storage_context_->getStorage(key);
      } else {
        int64_t numel = args.at(4).toInt();
        caffe2::TypeMeta dtype = at::CPU(type).typeMeta();

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

      auto options = at::CPU(type).options();
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
          device.is_hpu()) {
        tensor = tensor.to(device, tensor.scalar_type());
      } else if (device.type() != DeviceType::CPU) {
        AT_ERROR(
            "supported devices include CPU, CUDA and HPU, however got ",
            DeviceTypeName(device.type(), false));
      }
      stack_.emplace_back(std::move(tensor));
    } break;
    case PickleOpCode2::SETITEM: {
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
      stack_.erase(stack_.begin() + (key_pos), stack_.end());
    } break;
    default: {
      AT_ERROR(
          "Unknown opcode for unpickling at ",
          reinterpret_cast<void*>(opcode),
          ": ",
          int(static_cast<uint8_t>(opcode)));
    } break;
  }
  return opcode;
}

void LanternUnpickler::readGlobal(
    const std::string& module_name,
    const std::string& class_name) {
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
      AT_ERROR("Unknown pickler class id", class_name);
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
        AT_ERROR("Unknown pickler class id ", class_name);
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
    globals_.emplace_back([this] {
      // drop the Tuple that was argument to OrderedDict, and replace it
      // with None OrderedDicts only appear in tensor deserialization and
      // their value is never used
      //stack_.back() = IValue();
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
    c10::optional<c10::ScalarType> scalar_type;
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

    c10::optional<at::QScheme> qscheme;
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

void LanternUnpickler::rebuildSparseTensor() {
  globals_.emplace_back([this] {
    auto tup = pop(stack_).toTuple();
    const auto& elements = tup->elements();
    size_t idx = 0;
    auto layout = elements.at(idx++).toInt();
    at::Tensor result;
    switch (layout) {
      case static_cast<int>(c10::Layout::Sparse): {
        std::vector<int64_t> size = tupleToIntList(elements.at(idx++));
        bool requires_grad = elements.at(idx++).toBool();
        auto& indices_tensor = elements.at(idx++).toTensor();
        auto& values_tensor = elements.at(idx++).toTensor();
        auto options = values_tensor.options()
                           .layout(c10::Layout::Sparse)
                           .requires_grad(requires_grad);
        result = at::_sparse_coo_tensor_unsafe(
            indices_tensor, values_tensor, size, options);
        result = autograd::make_variable(result, options.requires_grad());
        break;
      }
      case static_cast<int>(c10::Layout::SparseCsr): {
        std::vector<int64_t> size = tupleToIntList(elements.at(idx++));
        bool requires_grad = elements.at(idx++).toBool();
        auto& crow_indices = elements.at(idx++).toTensor();
        auto& col_indices = elements.at(idx++).toTensor();
        auto& values_tensor = elements.at(idx++).toTensor();
        auto options = values_tensor.options()
                           .layout(c10::Layout::SparseCsr)
                           .requires_grad(requires_grad);
        result = at::_sparse_csr_tensor_unsafe(
            crow_indices, col_indices, values_tensor, size, options);
        result =
            autograd::make_variable(std::move(result), options.requires_grad());
        break;
      }
      default:
        TORCH_CHECK(
            false,
            "Unsupported sparse tensor layout type in serialization ",
            static_cast<c10::Layout>(layout));
        break;
    }
    stack_.emplace_back(std::move(result));
  });
}

void LanternUnpickler::rebuildTensor(bool quantized) {
  globals_.emplace_back([this, quantized] {
    auto tup = pop(stack_).toTuple();
    const auto& elements = tup->elements();
    size_t idx = 0;
    auto& storage_tensor = elements.at(idx++).toTensor();
    int64_t storage_offset = elements.at(idx++).toInt();
    std::vector<int64_t> size = tupleToIntList(elements.at(idx++));
    std::vector<int64_t> stride = tupleToIntList(elements.at(idx++));
    at::Tensor result;
    if (quantized) {
      auto qparams_tuple = elements.at(idx++).toTuple();
      const auto& qparams = qparams_tuple->elements();
      auto qscheme = static_cast<at::QScheme>(qparams.at(0).toInt());
      switch (qscheme) {
        case at::kPerTensorAffine: {
          double q_scale = qparams.at(1).toDouble();
          int64_t q_zero_point = qparams.at(2).toInt();
          result = at::_empty_affine_quantized(
              {0}, storage_tensor.options(), q_scale, q_zero_point);
        } break;
        case at::kPerChannelAffineFloatQParams:
        case at::kPerChannelAffine: {
          const auto& scales = qparams.at(1).toTensor();
          const auto& zero_points = qparams.at(2).toTensor();
          int64_t axis = qparams.at(3).toInt();
          result = at::_empty_per_channel_affine_quantized(
              {0}, scales, zero_points, axis, storage_tensor.options());
        } break;
        default:
          TORCH_CHECK(
              false,
              "Unsupported tensor quantization type in serialization ",
              toString(qscheme));
          break;
      }
    } else {
      result = at::empty({0}, storage_tensor.options());
    }
    bool requires_grad = elements.at(idx).toBool();
    // elements[idx++] is empty backwards hooks
    at::TensorImpl* impl = result.unsafeGetTensorImpl();
    impl->set_storage_keep_dtype(storage_tensor.storage());
    impl->set_storage_offset(storage_offset);
    impl->set_sizes_and_strides(size, stride);
    result = autograd::make_variable(result, requires_grad);
    stack_.emplace_back(std::move(result));
  });
}

#ifdef USE_RPC
void LanternUnpickler::rebuildRRef() {
  globals_.emplace_back([this] {
    // It is the same as how rref is unpickled in python,
    // see PyRRef::unpickle
    auto tuple = std::move(stack_.back()).toTuple();
    const auto& args = tuple->elements();
    stack_.pop_back();
    TORCH_INTERNAL_ASSERT(
        args.size() == distributed::rpc::RFD_TUPLE_SIZE,
        "Pickled RRefForkData must contain 7 numbers.");
    auto ownerId =
        static_cast<int16_t>(args.at(distributed::rpc::OWNER_IDX).toInt());
    // const reference will extend the lifetime of the temporary variable
    const auto& rrefId = distributed::rpc::RRefId(
        static_cast<int16_t>(args.at(distributed::rpc::RREFID_ON_IDX).toInt()),
        static_cast<int64_t>(args.at(distributed::rpc::RREFID_ID_IDX).toInt()));
    const auto& forkId = distributed::rpc::RRefId(
        static_cast<int16_t>(args.at(distributed::rpc::FORKID_ON_IDX).toInt()),
        static_cast<int64_t>(args.at(distributed::rpc::FORKID_ID_IDX).toInt()));
    auto parent =
        static_cast<int16_t>(args.at(distributed::rpc::PARENT_IDX).toInt());
    const auto& typeStr = static_cast<std::string>(
        args.at(distributed::rpc::TYPE_IDX).toStringRef());
    auto rrefForkData = distributed::rpc::RRefForkData(
        ownerId, rrefId, forkId, parent, typeStr);
    auto& ctx = distributed::rpc::RRefContext::getInstance();
    c10::intrusive_ptr<distributed::rpc::RRef> rref;
    TORCH_INTERNAL_ASSERT(
        type_resolver_ != nullptr, "type_resolver_ is nullptr.");
    at::StrongTypePtr type = type_resolver_(c10::QualifiedName(typeStr));
    rref = ctx.getOrCreateRRef(rrefForkData, type.type_);
    ctx.notifyOwnerAndParentOfFork(
        rrefForkData.forkId_, rrefForkData.parent_, rref);
    stack_.emplace_back(
        c10::static_intrusive_pointer_cast<c10::RRefInterface>(rref));
  });
  stack_.emplace_back(int64_t(globals_.size() - 1));
  return;
}
#endif

void LanternUnpickler::readSlowWithBuffer(char* dest, size_t sz) {
  // First, read any partial from buffer (may be 0).
  // We explicitly assume that sz > buffer_remaining_,
  // and that sz is never bigger than buffer_.size().
  AT_ASSERT(sz > buffer_remaining_);
  const size_t from_old_buf = buffer_remaining_;
  if (from_old_buf != 0) {
    memcpy(dest, buffer_.data() + buffer_pos_, from_old_buf);
  }
  const size_t needed = sz - from_old_buf;
  // Full read into the buffer. The calls here all explicitly
  // assume that one buffer will be enough for any sz.
  AT_ASSERT(sz <= buffer_.size());
  buffer_remaining_ = reader_(buffer_.data(), buffer_.size());
  if (buffer_remaining_ < needed) {
    AT_ERROR("Unexpected end of pickler archive.");
  }
  memcpy(dest + from_old_buf, buffer_.data(), needed);
  buffer_pos_ = needed; // assignment (0'ed from read)
  buffer_remaining_ -= needed;
}

// Read a number of bytes from the input stream
std::string LanternUnpickler::readBytes(size_t length) {
  std::string data;
  static const size_t kSmallString = 64;
  if (length <= buffer_remaining_) {
    // Fast-path: entirely in buffer.
    data.assign(buffer_.data() + buffer_pos_, length);
    buffer_pos_ += length;
    buffer_remaining_ -= length;
  } else if (length <= kSmallString) {
    // If the string is smallish, do a full buffer read,
    // and read out of that buffer.
    data.resize(length);
    readSlowWithBuffer(&data[0], length);
  } else {
    // Otherwise, for larger strings, read what we can from
    // the buffer, and then read directly to the destination.
    const size_t from_old_buf = buffer_remaining_;
    if (from_old_buf != 0) {
      data.reserve(length);
      data.append(buffer_.data() + buffer_pos_, from_old_buf);
    }
    data.resize(length);
    const size_t needed = length - from_old_buf;
    size_t nread = reader_(&data[from_old_buf], needed);
    if (nread != needed) {
      AT_ERROR("Unexpected end of pickler archive.");
    }
    buffer_remaining_ = 0;
    // buffer_pos_ has no meaning with buffer_remaining_ == 0.
  }
  return data;
}

// Pop all the list items off of the stack and append them to the list at
// the corresponding MARK
void LanternUnpickler::readList(IValue list_ivalue) {
  TORCH_CHECK(!marks_.empty(), "Parsing error: marks_ is empty");
  size_t start = marks_.back();
  marks_.pop_back();
  auto num_elements = stack_.size() - start;
  auto elements = c10::ArrayRef<IValue>(stack_).slice(start);
  if (list_ivalue.isIntList()) {
    auto list = std::move(list_ivalue).toIntList();
    list.reserve(num_elements);
    for (const auto& elem : elements) {
      list.emplace_back(elem.toInt());
    }
  } else if (list_ivalue.isTensorList()) {
    auto list = std::move(list_ivalue).toTensorList();
    list.reserve(num_elements);
    for (const auto& elem : elements) {
      list.emplace_back(elem.toTensor());
    }
  } else if (list_ivalue.isDoubleList()) {
    auto list = std::move(list_ivalue).toDoubleList();
    list.reserve(num_elements);
    for (const auto& elem : elements) {
      list.emplace_back(elem.toDouble());
    }
  } else if (list_ivalue.isBoolList()) {
    auto list = std::move(list_ivalue).toBoolList();
    list.reserve(num_elements);
    for (const auto& elem : elements) {
      list.push_back(elem.toBool());
    }
  } else if (list_ivalue.isList()) {
    auto list = std::move(list_ivalue).toList();
    list.reserve(num_elements);
    for (const auto& elem : elements) {
      list.emplace_back(elem);
    }
  } else {
    AT_ERROR("Unknown IValue list kind: ", list_ivalue.tagKind());
  }

  stack_.erase(stack_.begin() + start, stack_.end());
}

inline bool is_valid_python_id_char(char c) {
  return c == '_' || c == '.' || (c >= '0' && c <= '9') ||
      (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

// Read a newline terminated string
std::string LanternUnpickler::readString() {
  std::string ss;
  while (true) {
    auto* const bufferStart = buffer_.data() + buffer_pos_;
    const auto bufferLeft = buffer_.size() - buffer_pos_;
    char* const newlinePtr =
        static_cast<char*>(memchr(bufferStart, '\n', bufferLeft));
    if (newlinePtr) {
      // read up to newline and we are done.
      auto const charsRead = newlinePtr - bufferStart;
      ss.append(bufferStart, charsRead);
      buffer_remaining_ -= charsRead + 1;
      buffer_pos_ += charsRead + 1;
      break;
    } else {
      // read whole buffer, refill
      for (const char* p = bufferStart; p < bufferStart + bufferLeft; ++p) {
        // Simple check just in case there is no terminating '\n'
        TORCH_CHECK(
            is_valid_python_id_char(*p),
            "Found character '",
            int(uint8_t(*p)),
            "' in string, ",
            "strings must be qualified Python identifiers");
      }
      ss.append(bufferStart, bufferLeft);
      buffer_remaining_ = reader_(buffer_.data(), buffer_.size());
      buffer_pos_ = 0;
    }
  }
  return ss;
}

} // namespace jit
} // namespace torch

#include <caffe2/serialize/inline_container.h>
#include <torch/csrc/jit/serialization/import_read.h>

namespace torch {
namespace jit {

IValue readArchiveAndTensors(
    const std::string& archive_name,
    const std::string& pickle_prefix,
    const std::string& tensor_prefix,
    c10::optional<TypeResolver> type_resolver,
    c10::optional<ObjLoader> obj_loader,
    c10::optional<at::Device> device,
    caffe2::serialize::PyTorchStreamReader& stream_reader,
    c10::TypePtr (*type_parser)(const std::string&),
    std::shared_ptr<DeserializationStorageContext> storage_context) {
  std::string picklename = pickle_prefix + archive_name + ".pkl";
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

  std::string tensor_dir_path =
      (tensor_prefix.compare("") != 0) ? tensor_prefix : archive_name + "/";

  auto read_record = [&](const std::string& name) {
    std::string ss = tensor_dir_path + name;
    return std::get<0>(stream_reader.getRecord(ss));
  };

  LanternUnpickler unpickler(
      reader,
      type_resolver ? std::move(*type_resolver) : nullptr,
      obj_loader ? std::move(*obj_loader) : nullptr,
      std::move(read_record),
      device,
      false,
      type_parser,
      storage_context);
  unpickler.set_version(stream_reader.version());
  return unpickler.parse_ivalue();
}
}
}

#include <torch/csrc/jit/serialization/pickle.h>

#include <ATen/core/ivalue.h>
#include <caffe2/serialize/inline_container.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/jit/serialization/export.h>
#include <torch/csrc/jit/serialization/import_read.h>

namespace torch {
namespace jit {

void pickle(
    std::function<void(const char* data_start, size_t data_len)> writer,
    const IValue& ivalue,
    std::vector<at::Tensor>* tensor_table) {
  Pickler pickler(std::move(writer), tensor_table, nullptr, nullptr);
  pickler.protocol();
  pickler.pushIValue(ivalue);
  pickler.stop();
}

std::vector<char> pickle(
    const IValue& ivalue,
    std::vector<at::Tensor>* tensor_table) {
  std::vector<char> data;

  pickle(
      [&](const char* bytes, size_t len) {
        data.insert(data.end(), bytes, bytes + len);
      },
      ivalue,
      tensor_table);

  return data;
}

#ifndef C10_MOBILE
class VectorReader : public caffe2::serialize::ReadAdapterInterface {
 public:
  VectorReader(std::vector<char> data) : data_(std::move(data)) {}

  size_t size() const override {
    return data_.size();
  }

  size_t read(uint64_t pos, void* buf, size_t n, const char* what)
      const override {
    std::copy(
        data_.data() + pos,
        data_.data() + pos + n,
        reinterpret_cast<char*>(buf));
    return n;
  }

 private:
  std::vector<char> data_;
};
#endif

IValue pickle_load2(const std::vector<char>& data) {
  // Read in the pickle data
#ifndef C10_MOBILE
  caffe2::serialize::PyTorchStreamReader reader(
      std::make_unique<VectorReader>(data));

  return readArchiveAndTensors(
      "data",
      /*pickle_prefix=*/"",
      /*tensor_prefix=*/"",
      /*type_resolver=*/c10::nullopt,
      /*obj_loader=*/c10::nullopt,
      /*device=*/c10::nullopt,
      reader);
#else
  AT_ERROR(
      "pickle_load not supported on mobile "
      "(see https://github.com/pytorch/pytorch/pull/30108)");
#endif
};

IValue unpickle(
    std::function<size_t(char*, size_t)> reader,
    TypeResolver type_resolver,
    c10::ArrayRef<at::Tensor> tensor_table,
    c10::TypePtr (*type_parser)(const std::string&)) {
  LanternUnpickler unpickler(
      std::move(reader), std::move(type_resolver), tensor_table, type_parser);
  return unpickler.parse_ivalue();
}

IValue unpickle(
    const char* data,
    size_t size,
    TypeResolver type_resolver,
    c10::ArrayRef<at::Tensor> tensor_table,
    c10::TypePtr (*type_parser)(const std::string&)) {
  size_t bytes_read = 0;
  return unpickle(
      [&](char* buffer, size_t len) -> size_t {
        if (bytes_read >= size) {
          return 0;
        }
        len = std::min(size - bytes_read, len);
        // Copy len bytes into buffer
        const char* start = data + bytes_read;
        std::memcpy(buffer, start, len);
        bytes_read += len;
        return len;
      },
      std::move(type_resolver),
      tensor_table,
      type_parser);
}

} // namespace jit
} // namespace torch