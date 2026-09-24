#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "gemmx/kernels.hpp"
#include "gemmx/matrix.hpp"
#include "system_info.hpp"

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
  std::vector<std::size_t> sizes{128, 256, 512, 1024};
  std::vector<std::string> kernels; // empty = all registered kernels
  std::vector<std::string> dtypes{"float", "double"};
  int reps = 5;
  double min_sample_ms = 50.0;
  std::string out_dir = "results/raw";
  std::uint64_t seed = 42;
  std::string experiment = "adhoc";   // name of the experiment script
  std::string git_commit = "unknown"; // code version that produced the run
};

// ---------------------------------------------------------------------------
// Command-line parsing
// ---------------------------------------------------------------------------
std::vector<std::string> split(std::string_view s) {
  std::vector<std::string> out;
  std::size_t start = 0;
  while (true) {
    const std::size_t end = s.find(',', start);
    const std::size_t stop = (end == std::string_view::npos) ? s.size() : end;
    if (stop > start)
      out.emplace_back(s.substr(start, stop - start));
    if (end == std::string_view::npos)
      break;
    start = end + 1;
  }
  return out;
}

[[noreturn]] void usage_and_exit(int code) {
  std::cout
      << "usage: gemmx_benchmarks [options]\n"
         "  --sizes 128,256,512      square matrix sizes (default "
         "128,256,512,1024)\n"
         "  --kernels reference,...  kernels to run (default: all)\n"
         "  --dtypes float,double    element types (default: both)\n"
         "  --reps 5                 timed repetitions per configuration\n"
         "  --min-sample-ms 50       minimum duration of one timed sample\n"
         "  --out results/raw        output directory\n"
         "  --seed 42                seed for input matrices\n"
         "  --experiment NAME        experiment label (recorded in metadata)\n"
         "  --git-commit HASH        code version (recorded in metadata)\n";
  std::exit(code);
}

Options parse_args(int argc, char **argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string_view flag = argv[i];
    if (flag == "--help" || flag == "-h")
      usage_and_exit(0);
    if (i + 1 >= argc) {
      std::cerr << "missing value for " << flag << "\n";
      usage_and_exit(1);
    }
    const std::string value = argv[++i];
    if (flag == "--sizes") {
      o.sizes.clear();
      for (const auto &t : split(value))
        o.sizes.push_back(std::stoull(t));
    } else if (flag == "--kernels") {
      o.kernels = split(value);
    } else if (flag == "--dtypes") {
      o.dtypes = split(value);
    } else if (flag == "--reps") {
      o.reps = std::stoi(value);
    } else if (flag == "--min-sample-ms") {
      o.min_sample_ms = std::stod(value);
    } else if (flag == "--out") {
      o.out_dir = value;
    } else if (flag == "--seed") {
      o.seed = std::stoull(value);
    } else if (flag == "--experiment") {
      o.experiment = value;
    } else if (flag == "--git-commit") {
      o.git_commit = value;
    } else {
      std::cerr << "unknown flag " << flag << "\n";
      usage_and_exit(1);
    }
  }
  if (o.reps < 1) {
    std::cerr << "--reps must be >= 1\n";
    usage_and_exit(1);
  }
  for (std::size_t n : o.sizes) {
    if (n == 0) {
      std::cerr << "sizes must be > 0\n";
      usage_and_exit(1);
    }
  }
  for (const auto &d : o.dtypes) {
    if (d != "float" && d != "double") {
      std::cerr << "unknown dtype " << d << "\n";
      usage_and_exit(1);
    }
  }
  return o;
}

// ---------------------------------------------------------------------------
// Freivalds verification: checks C == A*B in O(n^2) instead of O(n^3).
// Pick a random vector x and compare C*x against A*(B*x).
// A wrong C passes only if its error happens to be (nearly) orthogonal to x.
// Tolerance: the per-element error bound gamma_K * (|A||B|)_ij, carried
// through the product with x, plus the rounding of the check itself.
// ---------------------------------------------------------------------------
template <typename T>
bool freivalds_check(const gemmx::Matrix<T> &A, const gemmx::Matrix<T> &B,
                     const gemmx::Matrix<T> &C, std::uint64_t seed) {
  const std::size_t M = A.rows(), K = A.cols(), N = B.cols();

  gemmx::Matrix<double> x(N, 1);
  gemmx::fill_random(x, seed);

  // bx = B*x and abs_bx = |B|*|x|, both accumulated in double.
  std::vector<double> bx(K, 0.0), abs_bx(K, 0.0);
  for (std::size_t k = 0; k < K; ++k) {
    for (std::size_t j = 0; j < N; ++j) {
      const double p = static_cast<double>(B(k, j)) * x(j, 0);
      bx[k] += p;
      abs_bx[k] += std::abs(p);
    }
  }

  const double eps = std::numeric_limits<T>::epsilon();
  const double factor = 2.0 * static_cast<double>(K + N) * eps;

  for (std::size_t i = 0; i < M; ++i) {
    double abx = 0.0, bound = 0.0, cx = 0.0;
    for (std::size_t k = 0; k < K; ++k) {
      const double a = static_cast<double>(A(i, k));
      abx += a * bx[k];
      bound += std::abs(a) * abs_bx[k];
    }
    for (std::size_t j = 0; j < N; ++j)
      cx += static_cast<double>(C(i, j)) * x(j, 0);
    if (!(std::abs(cx - abx) <= factor * bound))
      return false; // NaN fails too
  }
  return true;
}

// ---------------------------------------------------------------------------
// Timing helpers
// ---------------------------------------------------------------------------
double elapsed_ns(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::nano>(b - a).count();
}

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  const std::size_t m = v.size() / 2;
  return (v.size() % 2) ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

// ---------------------------------------------------------------------------
// Benchmark all selected kernels for one element type.
// Returns false if any kernel failed verification.
// ---------------------------------------------------------------------------
template <typename T>
bool run_dtype(const Options &o, std::string_view dtype,
               std::string_view run_id, std::ofstream &csv) {
  bool all_ok = true;

  for (const auto &k : gemmx::all_kernels<T>()) {
    if (!o.kernels.empty() && std::find(o.kernels.begin(), o.kernels.end(),
                                        k.name) == o.kernels.end()) {
      continue;
    }

    for (std::size_t n : o.sizes) {
      // Setup: allocation + data generation, never inside the timed region.
      gemmx::Matrix<T> A(n, n), B(n, n), C(n, n);
      gemmx::fill_random(A, o.seed);
      gemmx::fill_random(B, o.seed + 1);

      // 1) Warm-up call. It also measures one call for calibration and
      //    produces the output we verify.
      const auto t0 = Clock::now();
      k.fn(A, B, C);
      const auto t1 = Clock::now();
      const double once_ns = elapsed_ns(t0, t1);

      // 2) Correctness gate: never time a kernel that produced a wrong answer.
      if (!freivalds_check(A, B, C, o.seed + 2)) {
        std::cerr << "FAILED verification: " << k.name << " " << dtype
                  << " n=" << n << " (not timed)\n";
        all_ok = false;
        continue;
      }

      // 3) Calibration: repeat the call enough times that one timed sample
      //    lasts at least min_sample_ms, so fast runs aren't lost in timer
      //    noise.
      const double min_ns = o.min_sample_ms * 1e6;
      const std::size_t iters =
          (once_ns >= min_ns) ? 1
                              : static_cast<std::size_t>(
                                    std::ceil(min_ns / std::max(once_ns, 1.0)));

      // 4) Timed repetitions. Every sample is written to the CSV.
      const double flops = 2.0 * static_cast<double>(n) *
                           static_cast<double>(n) * static_cast<double>(n);
      std::vector<double> samples;
      for (int rep = 0; rep < o.reps; ++rep) {
        const auto s0 = Clock::now();
        for (std::size_t it = 0; it < iters; ++it)
          k.fn(A, B, C);
        const auto s1 = Clock::now();

        const double per_call_ns =
            elapsed_ns(s0, s1) / static_cast<double>(iters);
        const double gflops = flops / per_call_ns; // flop per ns == GFLOP/s
        samples.push_back(gflops);

        csv << run_id << ',' << k.name << ',' << dtype << ',' << n << ',' << n
            << ',' << n << ",," << 1 << ',' << rep << ',' << iters << ','
            << std::format("{:.0f}", per_call_ns) << ','
            << std::format("{:.4f}", gflops) << '\n';
      }
      csv.flush(); // a crash later can't lose finished configurations

      std::cout << std::format(
          "{:<12} {:<6} n={:<5} iters={:<5} median {:8.3f} GFLOP/s\n", k.name,
          dtype, n, iters, median(samples));
    }
  }
  return all_ok;
}

// ---------------------------------------------------------------------------
// Metadata sidecar (JSON)
// ---------------------------------------------------------------------------
std::string json_escape(std::string_view s) {
  std::string out;
  for (char c : s) {
    if (c == '"' || c == '\\')
      out += '\\';
    if (static_cast<unsigned char>(c) < 0x20)
      continue; // drop control chars
    out += c;
  }
  return out;
}

void write_metadata(const std::filesystem::path &path, std::string_view run_id,
                    const Options &o) {
  namespace si = gemmx::sysinfo;
  auto q = [](std::string_view s) { return "\"" + json_escape(s) + "\""; };

  std::string sizes;
  for (std::size_t i = 0; i < o.sizes.size(); ++i) {
    sizes += (i ? ", " : "") + std::to_string(o.sizes[i]);
  }

  std::ofstream f(path);
  f << "{\n"
    << "  \"run_id\": " << q(run_id) << ",\n"
    << "  \"experiment\": " << q(o.experiment) << ",\n"
    << "  \"git_commit\": " << q(o.git_commit) << ",\n"
    << "  \"cpu_model\": " << q(si::cpu_brand()) << ",\n"
    << "  \"logical_cpus\": " << std::thread::hardware_concurrency() << ",\n"
    << "  \"os\": " << q(si::os_string()) << ",\n"
    << "  \"compiler\": " << q(si::compiler_string()) << ",\n"
    << "  \"build_type\": " << q(si::build_type()) << ",\n"
    << "  \"cxx_flags\": " << q(si::cxx_flags()) << ",\n"
    << "  \"march_native\": " << (si::native_enabled() ? "true" : "false")
    << ",\n"
    << "  \"timer\": \"std::chrono::steady_clock\",\n"
    << "  \"timer_period_ns\": "
    << 1e9 * Clock::period::num / static_cast<double>(Clock::period::den)
    << ",\n"
    << "  \"sizes\": [" << sizes << "],\n"
    << "  \"reps\": " << o.reps << ",\n"
    << "  \"min_sample_ms\": " << o.min_sample_ms << ",\n"
    << "  \"seed\": " << o.seed << ",\n"
    << "  \"warmup\": \"1 untimed call per configuration (also used for "
       "calibration and verification)\",\n"
    << "  \"verification\": \"Freivalds check on warm-up output before "
       "timing\",\n"
    << "  \"allocation_in_timing\": false\n"
    << "}\n";
}

} // namespace

int main(int argc, char **argv) {
  const Options o = parse_args(argc, argv);

  namespace fs = std::filesystem;
  fs::create_directories(o.out_dir);

  // Run ID = UTC timestamp, e.g. 20260925-143012. If two runs start in the
  // same second, add a suffix (-2, -3, ...) instead of overwriting evidence.
  const std::string base_id =
      std::format("{:%Y%m%d-%H%M%S}", std::chrono::floor<std::chrono::seconds>(
                                          std::chrono::system_clock::now()));
  std::string run_id = base_id;
  for (int s = 2; fs::exists(fs::path(o.out_dir) / (run_id + ".csv")); ++s) {
    run_id = base_id + "-" + std::to_string(s);
  }

  const fs::path csv_path = fs::path(o.out_dir) / (run_id + ".csv");
  const fs::path meta_path = fs::path(o.out_dir) / (run_id + ".json");

  if (gemmx::sysinfo::build_type() != "Release") {
    std::cerr << "WARNING: build type is '" << gemmx::sysinfo::build_type()
              << "', not Release. Numbers are not research-grade.\n";
  }

  write_metadata(meta_path, run_id, o);

  std::ofstream csv(csv_path);
  if (!csv) {
    std::cerr << "cannot open " << csv_path << "\n";
    return 1;
  }
  csv << "run_id,kernel,dtype,m,k,n,block_size,threads,rep,iters,time_ns,"
         "gflops\n";

  bool ok = true;
  for (const auto &dt : o.dtypes) {
    if (dt == "float")
      ok = run_dtype<float>(o, "float", run_id, csv) && ok;
    if (dt == "double")
      ok = run_dtype<double>(o, "double", run_id, csv) && ok;
  }

  std::cout << "raw results: " << csv_path.string() << "\n"
            << "metadata:    " << meta_path.string() << "\n";
  return ok ? 0 : 2;
}
