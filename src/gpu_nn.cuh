// gpu_nn.cuh — CUDA kernel for batched MLP inference
//
// Performs:  out[i] = W2 * ReLU(W1 * in[i] + b1) + b2
// for i in [0, batch_size).
//
// Designed for:
//   • Input dim:  768  (12 piece-planes × 64 squares)
//   • Hidden dim: 256  (tunable at compile time)
//   • Output dim: 1    (centipawn score, tanh-normalised to [-1, 1])
//
// One CUDA thread block handles one position; threads within the block
// compute hidden-layer activations in parallel, using shared memory to
// avoid redundant global loads.
//
// Memory layout (column-major, like cuBLAS):
//   weights1: [hidden_size, input_size]
//   weights2: [1, hidden_size]
//
// TODO: Replace custom kernel with cuBLAS SGEMM for better occupancy
// TODO: Add INT8 quantisation path for inference on older GPUs
// TODO: Support dynamic batch sizes without recompilation

#pragma once
#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// ─── Configuration ────────────────────────────────────────────────────────────
constexpr int GPU_INPUT_SIZE  = 768;
constexpr int GPU_HIDDEN_SIZE = 256;
constexpr int GPU_OUTPUT_SIZE = 1;
constexpr int GPU_MAX_BATCH   = 256;

// ─── CUDA error check helper ──────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "[CUDA] %s (%s:%d)\n",                         \
                    cudaGetErrorString(err), __FILE__, __LINE__);           \
            std::abort();                                                   \
        }                                                                   \
    } while (0)

// ─── Kernel: one block per position ──────────────────────────────────────────
// Each thread computes one hidden neuron: dot(W1[n, :], in) + b1[n]
// Then thread 0 aggregates: out = dot(W2, hidden) + b2

__global__ void mlp_forward_kernel(
    const float* __restrict__ inputs,    // [batch, INPUT_SIZE]
    float*       __restrict__ outputs,   // [batch, 1]
    const float* __restrict__ w1,        // [HIDDEN, INPUT]
    const float* __restrict__ b1,        // [HIDDEN]
    const float* __restrict__ w2,        // [1, HIDDEN]
    const float* __restrict__ b2,        // [1]
    int input_size,
    int hidden_size,
    int batch_size
) {
    // Each block handles one sample
    int sample = blockIdx.x;
    if (sample >= batch_size) return;

    // Shared memory for hidden activations
    extern __shared__ float hidden[];  // [hidden_size]

    const float* in = inputs + sample * input_size;

    // Each thread computes one hidden neuron
    int tid = threadIdx.x;
    if (tid < hidden_size) {
        const float* row = w1 + tid * input_size;
        float acc = b1[tid];
        // Unrolled dot product
        for (int k = 0; k < input_size; k += 4) {
            acc += row[k]   * in[k];
            acc += row[k+1] * in[k+1];
            acc += row[k+2] * in[k+2];
            acc += row[k+3] * in[k+3];
        }
        // Handle remainder
        for (int k = (input_size / 4) * 4; k < input_size; ++k)
            acc += row[k] * in[k];
        // ReLU activation
        hidden[tid] = fmaxf(0.0f, acc);
    }
    __syncthreads();

    // Thread 0 computes the output neuron
    if (tid == 0) {
        float out = b2[0];
        for (int h = 0; h < hidden_size; ++h)
            out += w2[h] * hidden[h];
        // Tanh to map to [-1, 1]
        outputs[sample] = tanhf(out);
    }
}

// ─── GPU inference manager ────────────────────────────────────────────────────
// Holds device memory and provides a clean C++ interface.

class GpuInferenceEngine {
public:
    GpuInferenceEngine() = default;

    ~GpuInferenceEngine() { free_device(); }

    // Non-copyable
    GpuInferenceEngine(const GpuInferenceEngine&)            = delete;
    GpuInferenceEngine& operator=(const GpuInferenceEngine&) = delete;

    /// Upload weights to device.  Call once after loading the model.
    void upload_weights(const float* w1, const float* b1,
                        const float* w2, const float* b2,
                        int input_size, int hidden_size) {
        free_device();
        input_size_  = input_size;
        hidden_size_ = hidden_size;

        size_t w1_bytes = static_cast<size_t>(hidden_size) * input_size  * sizeof(float);
        size_t b1_bytes = static_cast<size_t>(hidden_size)               * sizeof(float);
        size_t w2_bytes = static_cast<size_t>(hidden_size)               * sizeof(float);
        size_t b2_bytes =                                                   sizeof(float);
        size_t in_bytes = static_cast<size_t>(GPU_MAX_BATCH) * input_size * sizeof(float);
        size_t out_bytes= static_cast<size_t>(GPU_MAX_BATCH)               * sizeof(float);

        CUDA_CHECK(cudaMalloc(&d_w1_,  w1_bytes));
        CUDA_CHECK(cudaMalloc(&d_b1_,  b1_bytes));
        CUDA_CHECK(cudaMalloc(&d_w2_,  w2_bytes));
        CUDA_CHECK(cudaMalloc(&d_b2_,  b2_bytes));
        CUDA_CHECK(cudaMalloc(&d_in_,  in_bytes));
        CUDA_CHECK(cudaMalloc(&d_out_, out_bytes));

        CUDA_CHECK(cudaMemcpy(d_w1_, w1, w1_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b1_, b1, b1_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_w2_, w2, w2_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b2_, b2, b2_bytes, cudaMemcpyHostToDevice));
    }

    /// Run batched inference.
    /// inputs:  [batch_size × input_size] on host
    /// outputs: [batch_size] on host (filled on return)
    void infer(const float* inputs, float* outputs, int batch_size) {
        assert(batch_size <= GPU_MAX_BATCH);
        assert(d_w1_ != nullptr && "Call upload_weights first");

        size_t in_bytes  = static_cast<size_t>(batch_size) * input_size_ * sizeof(float);
        size_t out_bytes = static_cast<size_t>(batch_size)               * sizeof(float);

        // Host → Device
        CUDA_CHECK(cudaMemcpyAsync(d_in_, inputs, in_bytes,
                                   cudaMemcpyHostToDevice, stream_));

        // Launch: one block per sample, hidden_size threads per block
        int shared_bytes = hidden_size_ * sizeof(float);
        mlp_forward_kernel<<<batch_size, hidden_size_, shared_bytes, stream_>>>(
            d_in_, d_out_,
            d_w1_, d_b1_, d_w2_, d_b2_,
            input_size_, hidden_size_, batch_size
        );
        CUDA_CHECK(cudaGetLastError());

        // Device → Host
        CUDA_CHECK(cudaMemcpyAsync(outputs, d_out_, out_bytes,
                                   cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

    bool is_ready() const { return d_w1_ != nullptr; }

private:
    float* d_w1_{nullptr};
    float* d_b1_{nullptr};
    float* d_w2_{nullptr};
    float* d_b2_{nullptr};
    float* d_in_{nullptr};
    float* d_out_{nullptr};
    int input_size_{GPU_INPUT_SIZE};
    int hidden_size_{GPU_HIDDEN_SIZE};
    cudaStream_t stream_{0};  // Default stream; create a dedicated one for production

    void free_device() {
        if (d_w1_)  { cudaFree(d_w1_);  d_w1_  = nullptr; }
        if (d_b1_)  { cudaFree(d_b1_);  d_b1_  = nullptr; }
        if (d_w2_)  { cudaFree(d_w2_);  d_w2_  = nullptr; }
        if (d_b2_)  { cudaFree(d_b2_);  d_b2_  = nullptr; }
        if (d_in_)  { cudaFree(d_in_);  d_in_  = nullptr; }
        if (d_out_) { cudaFree(d_out_); d_out_ = nullptr; }
    }
};

// ─── C-linkage wrapper (called from evaluator.cpp) ───────────────────────────
extern "C" {
void gpu_batch_infer(const float* inputs, float* outputs,
                     int batch_size, int input_size,
                     const float* weights1, const float* biases1,
                     int hidden1,
                     const float* weights2, const float* biases2) {
    static GpuInferenceEngine engine;
    if (!engine.is_ready())
        engine.upload_weights(weights1, biases1, weights2, biases2,
                               input_size, hidden1);
    engine.infer(inputs, outputs, batch_size);
}
}  // extern "C"

#endif  // USE_CUDA
