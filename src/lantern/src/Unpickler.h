#include <torch/csrc/jit/serialization/unpickler.h>

#ifdef _WIN32
#define LANTERN_API __declspec(dllexport)
#else
#define LANTERN_API
#endif

namespace torch {
namespace jit {

LANTERN_API IValue lantern_read_pickle(
    const std::string& archive_name,
    caffe2::serialize::PyTorchStreamReader& stream_reader);

} // namespace jit
} // namespace torch