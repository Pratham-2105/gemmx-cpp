#pragma once

#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>
#include <string>

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#endif

// Injected by CMake (see CMakeLists.txt). Fallbacks for builds outside CMake.
#ifndef GEMMX_BUILD_TYPE
#define GEMMX_BUILD_TYPE "unknown"
#endif
#ifndef GEMMX_CXX_FLAGS
#define GEMMX_CXX_FLAGS "unknown"
#endif
#ifndef GEMMX_NATIVE_ENABLED
#define GEMMX_NATIVE_ENABLED 0
#endif

namespace gemmx::sysinfo {

// CPU brand string, e.g. "13th Gen Intel(R) Core(TM) i5-13450HX",
// read straight from the CPU via the CPUID instruction (leaves 0x80000002-4).
inline std::string cpu_brand() {
#if defined(_MSC_VER) || defined(__x86_64__) || defined(__i386__)
  unsigned int regs[12] = {};
#if defined(_MSC_VER)
  int r[4];
  __cpuid(r, static_cast<int>(0x80000000u));
  if (static_cast<unsigned int>(r[0]) < 0x80000004u)
    return "unknown";
  for (int i = 0; i < 3; ++i) {
    __cpuid(r, static_cast<int>(0x80000002u + i));
    std::memcpy(&regs[4 * i], r, sizeof r);
  }
#else
  if (__get_cpuid_max(0x80000000u, nullptr) < 0x80000004u)
    return "unknown";
  for (unsigned int i = 0; i < 3; ++i) {
    __get_cpuid(0x80000002u + i, &regs[4 * i], &regs[4 * i + 1],
                &regs[4 * i + 2], &regs[4 * i + 3]);
  }
#endif
  char buf[49] = {};
  std::memcpy(buf, regs, 48);
  const std::string s(buf);
  const auto first = s.find_first_not_of(' ');
  if (first == std::string::npos)
    return "unknown";
  const auto last = s.find_last_not_of(' ');
  return s.substr(first, last - first + 1);
#else
  return "unknown (non-x86)";
#endif
}

inline std::string compiler_string() {
#if defined(__clang__)
  return "clang " + std::to_string(__clang_major__) + "." +
         std::to_string(__clang_minor__) + "." +
         std::to_string(__clang_patchlevel__);
#elif defined(__GNUC__)
  return "gcc " + std::to_string(__GNUC__) + "." +
         std::to_string(__GNUC_MINOR__) + "." +
         std::to_string(__GNUC_PATCHLEVEL__);
#elif defined(_MSC_VER)
  return "msvc " + std::to_string(_MSC_FULL_VER);
#else
  return "unknown";
#endif
}

// "linux-wsl <kernel release>", "linux <kernel release>", or "windows".
inline std::string os_string() {
#if defined(_WIN32)
  return "windows";
#elif defined(__linux__)
  std::ifstream f("/proc/sys/kernel/osrelease");
  std::string release;
  std::getline(f, release);
  std::string lower = release;
  std::transform(
      lower.begin(), lower.end(), lower.begin(),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  const bool wsl = lower.find("microsoft") != std::string::npos;
  return (wsl ? "linux-wsl " : "linux ") + release;
#else
  return "unknown";
#endif
}

inline std::string build_type() { return GEMMX_BUILD_TYPE; }
inline std::string cxx_flags() { return GEMMX_CXX_FLAGS; }
inline bool native_enabled() { return GEMMX_NATIVE_ENABLED != 0; }

} // namespace gemmx::sysinfo
