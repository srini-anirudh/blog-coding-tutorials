"""Small CUDA C kernels, compiled at run time with NVRTC through torch.cuda._compile_kernel.

Each kernel isolates one mechanism from the post. They are written for clarity, not speed.
Kernel arguments: tensors become pointers, Python ints become C int, Python floats become C double.
"""

import ctypes
import functools

import torch

SOURCES = {
    # One thread runs a chain of n dependent FMAs and reads the SM cycle counter around it.
    "fma_chain": r"""
extern "C" __global__ void fma_chain(float* out, long long* cyc, int n, double a) {
  float x = threadIdx.x, fa = (float)a;
  long long t0 = clock64();
  #pragma unroll 16
  for (int i = 0; i < n; ++i) x = fmaf(x, fa, 0.5f);   // unrolled: GPUs do not predict branches
  long long t1 = clock64();
  out[threadIdx.x] = x; cyc[threadIdx.x] = t1 - t0;
}""",
    # Every thread runs one dependent chain. Throughput depends only on how many threads are in flight.
    "fma_threads": r"""
extern "C" __global__ void fma_threads(float* out, int n, int total, double a) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= total) return;
  float x = t, fa = (float)a;
  #pragma unroll 16
  for (int i = 0; i < n; ++i) x = fmaf(x, fa, 0.5f);
  out[t] = x;
}""",
    # Pointer chase: each load's address is the previous load's value, so nothing overlaps.
    # LOAD is a plain load (may hit L1) or __ldcg (skips L1, so the fastest level is L2).
    "chase": r"""
extern "C" __global__ void chase(const unsigned* next, int steps, long long* cyc, unsigned* sink) {
  unsigned p = 0;
  for (int i = 0; i < 64; ++i) p = LOAD(&next[p]);   // warm the path
  long long t0 = clock64();
  #pragma unroll 1
  for (int i = 0; i < steps; ++i) p = LOAD(&next[p]);
  long long t1 = clock64();
  sink[0] = p; cyc[0] = t1 - t0;
}""",
    # Each thread takes the path path[tid]. The asm strings differ, so the compiler cannot merge the bodies.
    "diverge": r"""
#define BODY(K) case K: _Pragma("unroll 16") for (int i = 0; i < iters; ++i) \
    asm volatile("fma.rn.f32 %0, %0, %1, %2; // path " #K : "+f"(x) : "f"(fa), "f"(fb)); break;
extern "C" __global__ void diverge(float* out, const int* path, int iters, double a) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  float x = t, fa = (float)a, fb = 0.5f;
  switch (path[t]) { BODY(0) BODY(1) BODY(2) BODY(3) BODY(4) BODY(5) BODY(6) BODY(7) }
  out[t] = x;
}""",
    # A loop whose trip count is chosen per thread.
    "trips": r"""
extern "C" __global__ void trips(float* out, const int* niter, double a) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  float x = t, fa = (float)a;
  int n = niter[t];
  #pragma unroll 16
  for (int i = 0; i < n; ++i) x = fmaf(x, fa, 0.5f);
  out[t] = x;
}""",
    # Streaming read, ILP independent 16-byte loads in flight per thread per iteration.
    "stream": r"""
extern "C" __global__ void stream(const float4* in, float* out, int n4) {
  int t = blockIdx.x * blockDim.x + threadIdx.x, step = gridDim.x * blockDim.x;
  float s = 0.f;
  for (int i = t; i + (ILP - 1) * step < n4; i += ILP * step) {
    float4 v[ILP];
    #pragma unroll
    for (int j = 0; j < ILP; ++j) v[j] = __ldcs(&in[i + j * step]);
    #pragma unroll
    for (int j = 0; j < ILP; ++j) s += v[j].x + v[j].y + v[j].z + v[j].w;
  }
  out[t] = s;
}""",
    # Thread i reads in[i * stride]: consecutive threads are `stride` words apart.
    "strided": r"""
extern "C" __global__ void strided(const float* in, float* out, int stride, int n) {
  int t = blockIdx.x * blockDim.x + threadIdx.x, step = gridDim.x * blockDim.x;
  float s = 0.f;
  #pragma unroll 8
  for (int i = t; i < n; i += step) s += __ldcs(&in[(long long)i * stride]);  // unrolled: 8 loads in flight
  out[t] = s;
}""",
    # Thread i reads in[idx[i]].
    "gather": r"""
extern "C" __global__ void gather(const float* in, const int* idx, float* out, int n) {
  int t = blockIdx.x * blockDim.x + threadIdx.x, step = gridDim.x * blockDim.x;
  float s = 0.f;
  #pragma unroll 8
  for (int i = t; i < n; i += step) s += __ldcs(&in[idx[i]]);
  out[t] = s;
}""",
    # Lane l reads shared word addr[l], reps x 8 times. Thread 0 of each block records the block's cycles.
    "banks": r"""
extern "C" __global__ void banks(const int* addr, float* out, long long* cyc, int reps) {
  __shared__ float tile[2048];
  for (int i = threadIdx.x; i < 2048; i += blockDim.x) tile[i] = i;
  volatile float* v = tile;
  int a = addr[threadIdx.x & 31];
  float s = 0.f;
  __syncthreads();
  long long t0 = clock64();
  #pragma unroll 1
  for (int r = 0; r < reps; ++r) {
    s += v[a]; s += v[a]; s += v[a]; s += v[a]; s += v[a]; s += v[a]; s += v[a]; s += v[a];
  }
  __syncthreads();
  long long t1 = clock64();
  out[blockIdx.x * blockDim.x + threadIdx.x] = s;
  if (threadIdx.x == 0) cyc[blockIdx.x] = t1 - t0;
}""",
    # One thread per C element, operands straight from global memory.
    "mm_naive": r"""
extern "C" __global__ void mm_naive(const float* A, const float* B, float* C, int n) {
  int r = blockIdx.y * blockDim.y + threadIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
  float acc = 0.f;
  for (int k = 0; k < n; ++k) acc += A[r * n + k] * B[k * n + c];
  C[r * n + c] = acc;
}""",
    # The same work, staged through T x T shared-memory tiles. Each loaded element is used T times.
    "mm_tiled": r"""
extern "C" __global__ void mm_tiled(const float* A, const float* B, float* C, int n) {
  __shared__ float As[T][T], Bs[T][T + 1];
  int tr = threadIdx.y, tc = threadIdx.x;
  int r = blockIdx.y * T + tr, c = blockIdx.x * T + tc;
  float acc = 0.f;
  for (int k0 = 0; k0 < n; k0 += T) {
    As[tr][tc] = A[r * n + k0 + tc];
    Bs[tr][tc] = B[(k0 + tr) * n + c];
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < T; ++k) acc += As[tr][k] * Bs[k][tc];
    __syncthreads();
  }
  C[r * n + c] = acc;
}""",
    # Many live accumulators, so the register count is set by --maxrregcount. Used for occupancy checks.
    "hungry": r"""
extern "C" __global__ void hungry(float* out, int n) {
  float acc[96];
  #pragma unroll
  for (int i = 0; i < 96; ++i) acc[i] = threadIdx.x * i;
  for (int k = 0; k < n; ++k) {
    #pragma unroll
    for (int i = 0; i < 96; ++i) acc[i] = fmaf(acc[i], acc[(i + 1) % 96], 1.0f);
  }
  float s = 0.f;
  #pragma unroll
  for (int i = 0; i < 96; ++i) s += acc[i];
  out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}""",
}


def source(name: str, **defines) -> str:
    head = "".join(f"#define {k} {v}\n" for k, v in defines.items())
    return head + SOURCES[name]


def compile_offline(name: str, cc: str = "90", options: list | None = None, **defines) -> bytes:
    """Compile with NVRTC without loading. Needs no GPU, so the tests can check every kernel compiles."""
    from torch.cuda._utils import _nvrtc_compile
    code, _ = _nvrtc_compile(source(name, **defines), name, cc, None, options)
    return code


@functools.cache
def _compile(name: str, options: tuple, defines: tuple):
    return torch.cuda._compile_kernel(source(name, **dict(defines)), name, nvcc_options=list(options) or None)


def get(name: str, options: tuple = (), **defines):
    return _compile(name, tuple(options), tuple(sorted(defines.items())))


# Driver queries (libcuda through ctypes) -------------------------------------------------------------------------

def _lib():
    from torch.cuda._utils import _get_gpu_runtime_library
    return _get_gpu_runtime_library()


def num_regs(kernel) -> int:
    v = ctypes.c_int()
    _lib().cuFuncGetAttribute(ctypes.byref(v), 4, kernel.func)  # CU_FUNC_ATTRIBUTE_NUM_REGS
    return v.value


def driver_blocks_per_sm(kernel, threads_per_block: int, dyn_smem: int) -> int:
    """cuOccupancyMaxActiveBlocksPerMultiprocessor: the driver's own occupancy calculator."""
    if dyn_smem >= 48 * 1024:
        kernel.set_shared_memory_config(dyn_smem)
    v = ctypes.c_int()
    rc = _lib().cuOccupancyMaxActiveBlocksPerMultiprocessor(
        ctypes.byref(v), kernel.func, threads_per_block, ctypes.c_size_t(dyn_smem))
    if rc != 0:
        raise RuntimeError(f"occupancy query failed with CUDA error {rc}")
    return v.value


def sm_clock_hz() -> float:
    """Maximum SM clock reported by the driver (CU_DEVICE_ATTRIBUTE_CLOCK_RATE, in kHz)."""
    v = ctypes.c_int()
    _lib().cuDeviceGetAttribute(ctypes.byref(v), 13, torch.cuda.current_device())
    return v.value * 1e3
