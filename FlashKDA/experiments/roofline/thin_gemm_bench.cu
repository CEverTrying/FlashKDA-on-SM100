#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CUDA_CHECK(expr)                                                        \
  do {                                                                          \
    cudaError_t status_ = (expr);                                                \
    if (status_ != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                   cudaGetErrorString(status_));                                 \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                           \
  } while (0)

#define CUBLAS_CHECK(expr)                                                       \
  do {                                                                          \
    cublasStatus_t status_ = (expr);                                             \
    if (status_ != CUBLAS_STATUS_SUCCESS) {                                      \
      std::fprintf(stderr, "cuBLAS error at %s:%d: status=%d\n", __FILE__,      \
                   __LINE__, static_cast<int>(status_));                         \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                           \
  } while (0)

namespace {

constexpr int kN = 6288;
constexpr int kK = 7168;
constexpr int kMs[] = {1, 8, 16, 64, 256, 1024, 4096, 16384, 65536};
constexpr int kWarmup = 10;
constexpr int kRepeats = 5;
constexpr float kTargetMs = 150.0f;

struct Measurement {
  int m;
  int iterations;
  double us;
  double ai;
  double tflops;
  double effective_gbps;
  double compute_roof_pct;
  double memory_roof_pct;
  double active_roof_tflops;
  double active_roof_pct;
};

double median(std::vector<float> values) {
  std::sort(values.begin(), values.end());
  const size_t middle = values.size() / 2;
  if (values.size() % 2 != 0) return values[middle];
  return 0.5 * (values[middle - 1] + values[middle]);
}

void print_usage(const char* program) {
  std::fprintf(stderr,
               "Usage: %s [peak_bf16_tflops] [peak_hbm_gbps] [csv_path]\n",
               program);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc > 4) {
    print_usage(argv[0]);
    return EXIT_FAILURE;
  }

  const double peak_tflops = argc >= 2 ? std::atof(argv[1]) : 2250.0;
  const double peak_gbps = argc >= 3 ? std::atof(argv[2]) : 8000.0;
  const char* csv_path = argc >= 4 ? argv[3] : nullptr;
  if (!(peak_tflops > 0.0) || !(peak_gbps > 0.0)) {
    print_usage(argv[0]);
    return EXIT_FAILURE;
  }

  int device = 0;
  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaGetDeviceProperties(&properties, device));

  const int max_m = kMs[sizeof(kMs) / sizeof(kMs[0]) - 1];
  const size_t a_elements = static_cast<size_t>(max_m) * kK;
  const size_t w_elements = static_cast<size_t>(kN) * kK;
  const size_t d_elements = static_cast<size_t>(max_m) * kN;

  __nv_bfloat16* a = nullptr;
  __nv_bfloat16* w = nullptr;
  __nv_bfloat16* d = nullptr;
  CUDA_CHECK(cudaMalloc(&a, a_elements * sizeof(*a)));
  CUDA_CHECK(cudaMalloc(&w, w_elements * sizeof(*w)));
  CUDA_CHECK(cudaMalloc(&d, d_elements * sizeof(*d)));
  CUDA_CHECK(cudaMemset(a, 0, a_elements * sizeof(*a)));
  CUDA_CHECK(cudaMemset(w, 0, w_elements * sizeof(*w)));
  CUDA_CHECK(cudaMemset(d, 0, d_elements * sizeof(*d)));

  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
  const float alpha = 1.0f;
  const float beta = 0.0f;

  auto launch = [&](int m) {
    // Row-major D[M,N] = A[M,K] * W[N,K]^T is the same memory operation as
    // column-major D^T[N,M] = W_col[K,N]^T * A_col[K,M].
    CUBLAS_CHECK(cublasGemmEx(
        handle, CUBLAS_OP_T, CUBLAS_OP_N, kN, m, kK, &alpha, w, CUDA_R_16BF,
        kK, a, CUDA_R_16BF, kK, &beta, d, CUDA_R_16BF, kN,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
  };

  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  std::vector<Measurement> measurements;
  for (int m : kMs) {
    for (int i = 0; i < kWarmup; ++i) launch(m);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    launch(m);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float calibration_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&calibration_ms, start, stop));
    int iterations = static_cast<int>(std::ceil(kTargetMs /
                                                std::max(calibration_ms, 0.001f)));
    iterations = std::clamp(iterations, 5, 5000);

    std::vector<float> samples;
    samples.reserve(kRepeats);
    for (int repeat = 0; repeat < kRepeats; ++repeat) {
      CUDA_CHECK(cudaEventRecord(start));
      for (int i = 0; i < iterations; ++i) launch(m);
      CUDA_CHECK(cudaEventRecord(stop));
      CUDA_CHECK(cudaEventSynchronize(stop));
      float elapsed_ms = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
      samples.push_back(elapsed_ms / iterations);
    }

    const double ms = median(samples);
    const double flops = 2.0 * m * kN * kK;
    // Logical traffic model from assignment 4.5. This is effective bandwidth,
    // not a hardware-counter measurement of HBM traffic.
    const double bytes = 2.0 * (static_cast<double>(m) * kK +
                                static_cast<double>(kN) * kK +
                                static_cast<double>(m) * kN);
    const double ai = flops / bytes;
    const double tflops = flops / (ms * 1.0e9);
    const double effective_gbps = bytes / (ms * 1.0e6);
    const double memory_roof_tflops = ai * peak_gbps / 1000.0;
    const double active_roof_tflops = std::min(peak_tflops, memory_roof_tflops);
    measurements.push_back({
        m,
        iterations,
        ms * 1000.0,
        ai,
        tflops,
        effective_gbps,
        100.0 * tflops / peak_tflops,
        100.0 * tflops / memory_roof_tflops,
        active_roof_tflops,
        100.0 * tflops / active_roof_tflops,
    });
  }

  std::printf("device=%s cc=%d.%d N=%d K=%d peak=%.1f_TFLOPS bandwidth=%.1f_GB/s "
              "balance=%.2f_FLOP/byte\n",
              properties.name, properties.major, properties.minor, kN, kK,
              peak_tflops, peak_gbps, peak_tflops * 1000.0 / peak_gbps);
  std::printf("GB/s is effective logical traffic from the assignment 4.5 model.\n");
  std::printf("%7s %7s %10s %10s %11s %11s %12s %12s %11s\n", "M", "iters",
              "us", "AI", "TFLOPS", "eff.GB/s", "%compute", "%memory",
              "%active");
  for (const auto& x : measurements) {
    std::printf("%7d %7d %10.2f %10.2f %11.2f %11.2f %11.2f%% %11.2f%% %10.2f%%\n",
                x.m, x.iterations, x.us, x.ai, x.tflops, x.effective_gbps,
                x.compute_roof_pct, x.memory_roof_pct, x.active_roof_pct);
  }

  if (csv_path != nullptr && std::strlen(csv_path) > 0) {
    std::FILE* csv = std::fopen(csv_path, "w");
    if (csv == nullptr) {
      std::perror(csv_path);
      return EXIT_FAILURE;
    }
    std::fprintf(csv,
                 "layer,M,N,K,iters,us,AI_FLOP_per_byte,TFLOPS,effective_GBps,"
                 "compute_roof_pct,memory_roof_pct,active_roof_TFLOPS,active_roof_pct\n");
    for (const auto& x : measurements) {
      std::fprintf(csv,
                   "in_proj_qkvgfab,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,"
                   "%.6f,%.6f,%.6f\n",
                   x.m, kN, kK, x.iterations, x.us, x.ai, x.tflops,
                   x.effective_gbps, x.compute_roof_pct, x.memory_roof_pct,
                   x.active_roof_tflops, x.active_roof_pct);
    }
    std::fclose(csv);
    std::printf("csv=%s\n", csv_path);
  }

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUBLAS_CHECK(cublasDestroy(handle));
  CUDA_CHECK(cudaFree(a));
  CUDA_CHECK(cudaFree(w));
  CUDA_CHECK(cudaFree(d));
  return EXIT_SUCCESS;
}
