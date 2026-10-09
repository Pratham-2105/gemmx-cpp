#pragma once
// Runtime check for AVX2 + FMA support (GCC/Clang on x86-64, e.g. WSL2).

#include <cpuid.h>
#include <cstdint>

namespace gemmx {

// Does the OS save/restore the 256-bit YMM registers on context switches?
// If not, using AVX would corrupt data. XGETBV reads that setting.
inline bool os_supports_ymm() {
  unsigned eax, ebx, ecx, edx;
  if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx))
    return false;

  const bool osxsave = (ecx & (1u << 27)) != 0; // XGETBV is usable
  const bool avx = (ecx & (1u << 28)) != 0;     // CPU has AVX
  if (!osxsave || !avx)
    return false;

  uint32_t xcr0_lo, xcr0_hi;
  __asm__ volatile("xgetbv" : "=a"(xcr0_lo), "=d"(xcr0_hi) : "c"(0));
  // bit 1 = SSE state, bit 2 = AVX (YMM) state; both must be enabled
  return (xcr0_lo & 0x6u) == 0x6u;
}

inline bool cpu_has_fma() {
  unsigned eax, ebx, ecx, edx;
  if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx))
    return false;
  return (ecx & (1u << 12)) != 0 && os_supports_ymm();
}

inline bool cpu_has_avx2() {
  unsigned eax, ebx, ecx, edx;
  if (__get_cpuid_max(0, nullptr) < 7)
    return false;
  __cpuid_count(7, 0, eax, ebx, ecx, edx);
  return (ebx & (1u << 5)) != 0 && os_supports_ymm();
}

// What the kernel actually needs: both together.
inline bool cpu_has_avx2_fma() {
  static const bool ok = cpu_has_avx2() && cpu_has_fma();
  return ok;
}

} // namespace gemmx
