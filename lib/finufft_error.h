// Human-readable messages for FINUFFT/cuFINUFFT integer return codes.

#ifndef _JAX_FINUFFT_ERROR_H_
#define _JAX_FINUFFT_ERROR_H_

#include <finufft_common/safe_call.h>

#include <string>

namespace jax_finufft {

/// Describe a failed FINUFFT call, e.g. "FINUFFT makeplan failed: requested
/// tolerance too small (code 26)". `what` names the library and the call.
inline std::string error_message(const char* what, int code) {
  return std::string(what) + " failed: " + finufft::exception::error_string(code) + " (code " +
         std::to_string(code) + ")";
}

}  // namespace jax_finufft

#endif
