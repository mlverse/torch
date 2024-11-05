#include <torch/csrc/jit/serialization/unpickler.h>


namespace torch {
namespace jit {

class LanternUnpickler : public torch::jit::Unpickler {
public:
    void readGlobal(const std::string& module_name, const std::string& class_name);
    PickleOpCode readInstruction();
    void run();
    IValue parse_ivalue();
    using torch::jit::Unpickler::Unpickler;
};

IValue lantern_read_pickle(
    const std::string& archive_name,
    caffe2::serialize::PyTorchStreamReader& stream_reader);

} // namespace jit
} // namespace torch