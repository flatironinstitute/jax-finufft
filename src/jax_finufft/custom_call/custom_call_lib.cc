#include <nanobind/nanobind.h>

#include <string>

namespace nb = nanobind;
using namespace nb::literals;

namespace jax_finufft {
namespace custom_call {

struct Descriptor {
  void* function_ptr;
  std::string user_descriptor;
};

extern "C" void custom_call_wrapper(void* output, void** inputs) {
  auto descriptor = reinterpret_cast<Descriptor*>(*static_cast<uintptr_t*>(inputs[0]));
  inputs += 1;
  auto function_ptr =
      reinterpret_cast<void (*)(void*, size_t, void**, void**)>(descriptor->function_ptr);
  function_ptr(descriptor->user_descriptor.data(), descriptor->user_descriptor.size(), inputs,
               reinterpret_cast<void**>(output));
}

NB_MODULE(custom_call_lib, m) {
  m.def("get_custom_call_wrapper", []() {
    return nb::capsule(reinterpret_cast<void*>(&custom_call_wrapper), "xla._CUSTOM_CALL_TARGET");
  });
  m.def("build_descriptor", [](void* function_ptr, const std::string& user_descriptor) {
    Descriptor* descriptor = new Descriptor;
    descriptor->function_ptr = function_ptr;
    descriptor->user_descriptor = user_descriptor;
    uint64_t ptr = reinterpret_cast<uint64_t>(descriptor);
    nb::capsule keepalive = nb::capsule(
        descriptor, [](void* ptr) noexcept { delete reinterpret_cast<Descriptor*>(ptr); });
    return std::make_pair(ptr, keepalive);
  });
}

}  // namespace custom_call
}  // namespace jax_finufft