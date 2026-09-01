#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t error_ = (call);                                            \
        if (error_ != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,       \
                         __LINE__, cudaGetErrorString(error_));                  \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

namespace wmma = nvcuda::wmma;

constexpr int kTcgenM = 64;
constexpr int kPhysicalK = 64;

__host__ __device__ inline int swizzle_128b(int row, int column_byte) {
    int atom = row >> 3;
    int row_in_atom = row & 7;
    int chunk = column_byte >> 4;
    int byte_in_chunk = column_byte & 15;
    return atom * 1024 + row_in_atom * 128 +
           ((chunk ^ row_in_atom) << 4) + byte_in_chunk;
}

__device__ inline uint64_t make_smem_desc(uint32_t address) {
    uint64_t desc = 0;
    desc |= uint64_t((address >> 4) & 0x3fff);
    desc |= uint64_t(1024 >> 4) << 32;
    desc |= uint64_t(1) << 46;
    desc |= uint64_t(2) << 61;
    return desc;
}

__device__ inline uint32_t make_instruction_desc(int m, int n) {
    return (1u << 4) | (1u << 7) | (1u << 10) |
           (uint32_t(n >> 3) << 17) | (uint32_t(m >> 4) << 24);
}

__device__ inline void wait_mbarrier(uint32_t address, uint32_t phase) {
    uint32_t done = 0;
    while (!done) {
        asm volatile(
            "{\n.reg .pred p;\n"
            "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
            "selp.b32 %0, 1, 0, p;\n}"
            : "=r"(done)
            : "r"(address), "r"(phase));
    }
}

template <int Chunk>
__global__ void hmma_exact_kernel(const __nv_bfloat16* a,
                                  const __nv_bfloat16* b, float* output,
                                  int inner_iterations) {
    static_assert(Chunk == 16 || Chunk == 32);
    constexpr int tiles = Chunk / 16;
    __shared__ __align__(32) __nv_bfloat16 smem_a[Chunk * Chunk];
    __shared__ __align__(32) __nv_bfloat16 smem_b[Chunk * Chunk];
    for (int index = int(threadIdx.x); index < Chunk * Chunk;
         index += int(blockDim.x)) {
        smem_a[index] = a[index];
        smem_b[index] = b[index];
    }
    __syncthreads();

    int warp = int(threadIdx.x) >> 5;
    if (warp >= tiles * tiles) return;

    int tile_m = warp / tiles;
    int tile_n = warp % tiles;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16,
                   wmma::row_major>
        frag_a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16,
                   wmma::col_major>
        frag_b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
    wmma::fill_fragment(accumulator, 0.0f);

    for (int iteration = 0; iteration < inner_iterations; ++iteration) {
#pragma unroll
        for (int tile_k = 0; tile_k < tiles; ++tile_k) {
            wmma::load_matrix_sync(frag_a,
                                   smem_a + tile_m * 16 * Chunk + tile_k * 16,
                                   Chunk);
            // b is [N, K] row-major. The same bytes describe a column-major
            // [K, N] operand, so the operation computes A @ B^T.
            wmma::load_matrix_sync(frag_b,
                                   smem_b + tile_n * 16 * Chunk + tile_k * 16,
                                   Chunk);
            wmma::mma_sync(accumulator, frag_a, frag_b, accumulator);
        }
    }
    float* block_output = output + size_t(blockIdx.x) * Chunk * Chunk;
    wmma::store_matrix_sync(
        block_output + tile_m * 16 * Chunk + tile_n * 16, accumulator,
        Chunk, wmma::mem_row_major);
}

template <int Chunk>
__global__ void tcgen05_padded_kernel(const __nv_bfloat16* a,
                                      const __nv_bfloat16* b, float* output,
                                      int inner_iterations) {
    static_assert(Chunk == 16 || Chunk == 32);
    __shared__ __align__(1024) unsigned char smem_a[kTcgenM * kPhysicalK * 2];
    __shared__ __align__(1024) unsigned char smem_b[Chunk * kPhysicalK * 2];
    __shared__ __align__(8) uint64_t mbarrier;
    __shared__ uint32_t tmem_address_storage[1];

    int warp = int(threadIdx.x) >> 5;
    int lane = int(threadIdx.x) & 31;
    uint32_t mbarrier_address =
        uint32_t(__cvta_generic_to_shared(&mbarrier));
    if (lane == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::
                         "r"(mbarrier_address), "r"(1));
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    __syncthreads();

    uint32_t allocation_destination =
        uint32_t(__cvta_generic_to_shared(tmem_address_storage));
    if (warp == 0) {
        asm volatile(
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::
                "r"(allocation_destination), "r"(32));
        asm volatile(
            "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
    __syncthreads();

    for (int index = int(threadIdx.x); index < kTcgenM * kPhysicalK;
         index += int(blockDim.x)) {
        int row = index / kPhysicalK;
        int k = index % kPhysicalK;
        __nv_bfloat16 value = __float2bfloat16(0.0f);
        if (row < Chunk && k < Chunk) value = a[row * Chunk + k];
        *reinterpret_cast<__nv_bfloat16*>(
            smem_a + swizzle_128b(row, k * 2)) = value;
    }
    for (int index = int(threadIdx.x); index < Chunk * kPhysicalK;
         index += int(blockDim.x)) {
        int row = index / kPhysicalK;
        int k = index % kPhysicalK;
        __nv_bfloat16 value =
            k < Chunk ? b[row * Chunk + k] : __float2bfloat16(0.0f);
        *reinterpret_cast<__nv_bfloat16*>(
            smem_b + swizzle_128b(row, k * 2)) = value;
    }
    asm volatile("fence.proxy.async.shared::cta;");
    __syncthreads();

    uint32_t elected = 0;
    asm volatile(
        "{\n.reg .pred p;\nelect.sync _|p, 0xffffffff;\n"
        "selp.b32 %0, 1, 0, p;\n}"
        : "=r"(elected));
    uint32_t tmem_address = tmem_address_storage[0];
    if (warp == 0 && elected) {
        uint32_t a_base = uint32_t(__cvta_generic_to_shared(smem_a));
        uint32_t b_base = uint32_t(__cvta_generic_to_shared(smem_b));
        uint32_t instruction_desc = make_instruction_desc(kTcgenM, Chunk);
        asm volatile("tcgen05.fence::after_thread_sync;");
        for (int iteration = 0; iteration < inner_iterations; ++iteration) {
#pragma unroll
            for (int k = 0; k < Chunk; k += 16) {
                uint64_t desc_a = make_smem_desc(a_base + k * 2);
                uint64_t desc_b = make_smem_desc(b_base + k * 2);
                uint32_t accumulate = (iteration != 0 || k != 0);
                asm volatile(
                    "{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
                    "tcgen05.mma.cta_group::1.kind::f16 "
                    "[%0], %1, %2, %3, p;\n}\n" ::
                        "r"(tmem_address), "l"(desc_a), "l"(desc_b),
                        "r"(instruction_desc), "r"(accumulate)
                    : "memory");
            }
        }
        asm volatile(
            "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
            ".shared::cluster.b64 [%0];" ::
                "r"(mbarrier_address)
            : "memory");
    }
    wait_mbarrier(mbarrier_address, 0);
    asm volatile("tcgen05.fence::after_thread_sync;");

    float* block_output = output + size_t(blockIdx.x) * Chunk * Chunk;
    for (int column = 0; column < Chunk; column += 8) {
        // M=64 is interleaved over four 16-data-path subpartitions. Logical
        // rows [16w,16w+15] live at datapaths [32w,32w+15].
        uint32_t source =
            tmem_address + (uint32_t(warp * 32) << 16) + column;
        float values[8];
        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x8.b32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
            : "=f"(values[0]), "=f"(values[1]), "=f"(values[2]),
              "=f"(values[3]), "=f"(values[4]), "=f"(values[5]),
              "=f"(values[6]), "=f"(values[7])
            : "r"(source));
        asm volatile("tcgen05.wait::ld.sync.aligned;");
        int row = warp * 16 + lane;
        if (lane < 16 && row < Chunk) {
#pragma unroll
            for (int offset = 0; offset < 8; ++offset) {
                block_output[row * Chunk + column + offset] = values[offset];
            }
        }
    }
    __syncthreads();
    if (warp == 0) {
        asm volatile(
            "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::
                "r"(tmem_address), "r"(32));
    }
}

struct Timing {
    float milliseconds;
    bool correct;
    float max_abs_error;
};

template <int Chunk, class Launch>
Timing run_case(Launch launch, const std::vector<float>& reference,
                float* device_output, int blocks, int inner_iterations,
                int timing_iterations) {
    launch(blocks, inner_iterations);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> got(size_t(blocks) * Chunk * Chunk);
    CUDA_CHECK(cudaMemcpy(got.data(), device_output, got.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    float max_error = 0.0f;
    int printed_mismatches = 0;
    const int checked_blocks[] = {0, blocks / 2, blocks - 1};
    for (int block : checked_blocks) {
        size_t block_offset = size_t(block) * Chunk * Chunk;
        for (size_t i = 0; i < reference.size(); ++i) {
            float expected = reference[i] * inner_iterations;
            float actual = got[block_offset + i];
            max_error = std::max(max_error, std::fabs(actual - expected));
            if (actual != expected && printed_mismatches < 5) {
                std::fprintf(
                    stderr,
                    "mismatch chunk=%d block=%d row=%zu col=%zu got=%g want=%g\n",
                    Chunk, block, i / Chunk, i % Chunk, actual, expected);
                ++printed_mismatches;
            }
        }
    }

    for (int i = 0; i < 10; ++i) launch(blocks, inner_iterations);
    CUDA_CHECK(cudaGetLastError());
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < timing_iterations; ++i) {
        launch(blocks, inner_iterations);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return {elapsed_ms / timing_iterations, max_error == 0.0f, max_error};
}

template <int Chunk>
void benchmark_chunk(const __nv_bfloat16* device_a,
                     const __nv_bfloat16* device_b, float* device_output,
                     const std::vector<float>& reference, int blocks,
                     int inner_iterations, int timing_iterations,
                     const char* gpu_name, int compute_major,
                     int compute_minor) {
    auto hmma_launch = [&](int launch_blocks, int repetitions) {
        constexpr int threads = (Chunk / 16) * (Chunk / 16) * 32;
        hmma_exact_kernel<Chunk><<<launch_blocks, threads>>>(
            device_a, device_b, device_output, repetitions);
    };
    auto tcgen_launch = [&](int launch_blocks, int repetitions) {
        constexpr int threads = (Chunk / 16) * 32;
        tcgen05_padded_kernel<Chunk><<<launch_blocks, threads>>>(
            device_a, device_b, device_output, repetitions);
    };
    Timing hmma = run_case<Chunk>(hmma_launch, reference, device_output, blocks,
                                  inner_iterations, timing_iterations);
    Timing tcgen = run_case<Chunk>(tcgen_launch, reference, device_output,
                                   blocks, inner_iterations, timing_iterations);

    auto print_row = [&](const char* variant, int physical_m,
                         const Timing& timing) {
        double useful_flops = 2.0 * Chunk * Chunk * Chunk * blocks *
                              inner_iterations;
        double physical_flops =
            2.0 * physical_m * Chunk * Chunk * blocks * inner_iterations;
        double seconds = timing.milliseconds * 1.0e-3;
        std::printf(
            "%s,%s,%d.%d,%d,%d,%d,%d,%d,%d,%.6f,%d,%d,%.6f,%.6f,%.6f,%s,%.9g\n",
            variant, gpu_name, compute_major, compute_minor, Chunk, Chunk,
            Chunk, physical_m, Chunk, Chunk, double(Chunk) / physical_m,
            blocks, inner_iterations,
            timing.milliseconds * 1000.0, useful_flops / seconds / 1.0e12,
            physical_flops / seconds / 1.0e12,
            timing.correct ? "true" : "false", timing.max_abs_error);
    };
    print_row("hmma_exact", Chunk, hmma);
    print_row("tcgen05_m_padded", kTcgenM, tcgen);
}

int main(int argc, char** argv) {
    int inner_iterations = argc > 1 ? std::atoi(argv[1]) : 64;
    int timing_iterations = argc > 2 ? std::atoi(argv[2]) : 100;
    if (inner_iterations < 1 || timing_iterations < 1) {
        std::fprintf(stderr, "inner_iterations and timing_iterations must be positive\n");
        return 2;
    }

    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
    int blocks = properties.multiProcessorCount * 4;
    std::fprintf(stderr,
                 "GPU=%s SM=%d.%d SM_count=%d blocks=%d inner=%d timing=%d\n",
                 properties.name, properties.major, properties.minor,
                 properties.multiProcessorCount, blocks, inner_iterations,
                 timing_iterations);
    std::printf(
        "variant,gpu,compute_capability,chunk,logical_n,logical_k,physical_m,"
        "physical_n,physical_k,useful_fraction,blocks,inner_iterations,avg_us,"
        "useful_tflops,physical_tflops,correct,max_abs_error\n");

    std::mt19937 generator(42);
    std::uniform_int_distribution<int> distribution(-1, 1);
    for (int chunk : {16, 32}) {
        std::vector<__nv_bfloat16> host_a(chunk * chunk);
        std::vector<__nv_bfloat16> host_b(chunk * chunk);
        std::vector<float> reference(chunk * chunk, 0.0f);
        for (auto& value : host_a)
            value = __float2bfloat16(float(distribution(generator)));
        for (auto& value : host_b)
            value = __float2bfloat16(float(distribution(generator)));
        for (int m = 0; m < chunk; ++m) {
            for (int n = 0; n < chunk; ++n) {
                for (int k = 0; k < chunk; ++k) {
                    reference[m * chunk + n] +=
                        __bfloat162float(host_a[m * chunk + k]) *
                        __bfloat162float(host_b[n * chunk + k]);
                }
            }
        }

        __nv_bfloat16 *device_a = nullptr, *device_b = nullptr;
        float* device_output = nullptr;
        CUDA_CHECK(cudaMalloc(&device_a, host_a.size() * sizeof(*device_a)));
        CUDA_CHECK(cudaMalloc(&device_b, host_b.size() * sizeof(*device_b)));
        CUDA_CHECK(cudaMalloc(&device_output,
                              size_t(blocks) * chunk * chunk * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(device_a, host_a.data(),
                              host_a.size() * sizeof(*device_a),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(device_b, host_b.data(),
                              host_b.size() * sizeof(*device_b),
                              cudaMemcpyHostToDevice));
        if (chunk == 16) {
            benchmark_chunk<16>(device_a, device_b, device_output, reference,
                                blocks, inner_iterations, timing_iterations,
                                properties.name, properties.major,
                                properties.minor);
        } else {
            benchmark_chunk<32>(device_a, device_b, device_output, reference,
                                blocks, inner_iterations, timing_iterations,
                                properties.name, properties.major,
                                properties.minor);
        }
        CUDA_CHECK(cudaFree(device_a));
        CUDA_CHECK(cudaFree(device_b));
        CUDA_CHECK(cudaFree(device_output));
    }
    return 0;
}
