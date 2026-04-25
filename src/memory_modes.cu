#include <cuda_runtime.h>
#include <linux/mempolicy.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#define CHECK_CUDA(call) do { \
  cudaError_t err__ = (call); \
  if (err__ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
    exit(2); \
  } \
} while (0)

__global__ void rw_kernel(float *data, size_t n) {
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    data[i] = data[i] * 1.01f + 0.01f;
  }
}

static long page_size() {
  static long ps = sysconf(_SC_PAGESIZE);
  return ps;
}

static void bind_to_node_cpus(int node) {
  std::ifstream in("/sys/devices/system/node/node" + std::to_string(node) + "/cpulist");
  std::string text;
  std::getline(in, text);
  cpu_set_t set;
  CPU_ZERO(&set);
  std::stringstream ss(text);
  std::string part;
  while (std::getline(ss, part, ',')) {
    int start = 0, end = 0;
    if (sscanf(part.c_str(), "%d-%d", &start, &end) == 2) {
      for (int cpu = start; cpu <= end; ++cpu) CPU_SET(cpu, &set);
    } else if (sscanf(part.c_str(), "%d", &start) == 1) {
      CPU_SET(start, &set);
    }
  }
  sched_setaffinity(0, sizeof(set), &set);
}

static void *alloc_on_node(size_t bytes, int node) {
  void *ptr = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (ptr == MAP_FAILED) return nullptr;
  unsigned long nodemask = 1UL << node;
  long rc = syscall(SYS_mbind, ptr, bytes, MPOL_BIND, &nodemask, sizeof(nodemask) * 8, MPOL_MF_MOVE);
  if (rc != 0) {
    munmap(ptr, bytes);
    return nullptr;
  }
  return ptr;
}

static void touch_cpu(float *ptr, size_t bytes, int node) {
  bind_to_node_cpus(node);
  volatile char *p = reinterpret_cast<volatile char *>(ptr);
  for (size_t off = 0; off < bytes; off += page_size()) {
    p[off] = static_cast<char>(off);
  }
}

static int verify_pages(void *ptr, size_t bytes, int expected_node) {
  const size_t ps = page_size();
  size_t pages = (bytes + ps - 1) / ps;
  size_t samples = std::min<size_t>(pages, 4096);
  std::vector<void *> addrs(samples);
  std::vector<int> status(samples, -999);
  for (size_t i = 0; i < samples; ++i) {
    size_t page = (samples == pages) ? i : (i * pages / samples);
    addrs[i] = static_cast<char *>(ptr) + page * ps;
  }
  long rc = syscall(SYS_move_pages, 0, samples, addrs.data(), nullptr, status.data(), 0);
  if (rc != 0) return -1;
  int matched = 0;
  for (int node : status) {
    if (node == expected_node) matched++;
  }
  return matched;
}

static bool time_kernel_safe(float *device_ptr, size_t bytes, int iterations, std::vector<float> *gbs, std::string *error) {
  cudaDeviceProp prop{};
  int dev = 0;
  cudaError_t err = cudaGetDevice(&dev);
  if (err != cudaSuccess) {
    *error = cudaGetErrorString(err);
    return false;
  }
  err = cudaGetDeviceProperties(&prop, dev);
  if (err != cudaSuccess) {
    *error = cudaGetErrorString(err);
    return false;
  }
  int blocks = std::max(1, prop.multiProcessorCount * 4);
  int threads = 256;
  size_t n = bytes / sizeof(float);

  for (int i = 0; i < 3; ++i) {
    rw_kernel<<<blocks, threads>>>(device_ptr, n);
  }
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    *error = cudaGetErrorString(err);
    return false;
  }

  for (int i = 0; i < iterations; ++i) {
    cudaEvent_t start, stop;
    err = cudaEventCreate(&start);
    if (err != cudaSuccess) { *error = cudaGetErrorString(err); return false; }
    err = cudaEventCreate(&stop);
    if (err != cudaSuccess) { *error = cudaGetErrorString(err); cudaEventDestroy(start); return false; }
    err = cudaEventRecord(start);
    if (err != cudaSuccess) { *error = cudaGetErrorString(err); cudaEventDestroy(start); cudaEventDestroy(stop); return false; }
    rw_kernel<<<blocks, threads>>>(device_ptr, n);
    err = cudaEventRecord(stop);
    if (err != cudaSuccess) { *error = cudaGetErrorString(err); cudaEventDestroy(start); cudaEventDestroy(stop); return false; }
    err = cudaEventSynchronize(stop);
    if (err != cudaSuccess) { *error = cudaGetErrorString(err); cudaEventDestroy(start); cudaEventDestroy(stop); return false; }
    float ms = 0.0f;
    err = cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    if (err != cudaSuccess) { *error = cudaGetErrorString(err); return false; }
    double seconds = ms / 1000.0;
    gbs->push_back(static_cast<float>((2.0 * bytes) / seconds / 1e9));
  }
  return true;
}

static std::vector<float> time_kernel(float *device_ptr, size_t bytes, int iterations) {
  std::vector<float> gbs;
  std::string error;
  if (!time_kernel_safe(device_ptr, bytes, iterations, &gbs, &error)) {
    fprintf(stderr, "kernel timing failed: %s\n", error.c_str());
    exit(2);
  }
  return gbs;
}

static std::vector<float> time_kernel_alternating_cpu_gpu(float *ptr, size_t bytes, int iterations, int host_node) {
  cudaDeviceProp prop{};
  int dev = 0;
  CHECK_CUDA(cudaGetDevice(&dev));
  CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));
  int blocks = std::max(1, prop.multiProcessorCount * 4);
  int threads = 256;
  size_t n = bytes / sizeof(float);
  std::vector<float> gbs;
  for (int i = 0; i < iterations; ++i) {
    touch_cpu(ptr, bytes, host_node);
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    rw_kernel<<<blocks, threads>>>(ptr, n);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    gbs.push_back(static_cast<float>((2.0 * bytes) / (ms / 1000.0) / 1e9));
  }
  return gbs;
}

struct Stats {
  double min = 0.0;
  double median = 0.0;
  double max = 0.0;
  std::string notes;
};

static Stats summarize(std::vector<float> values, std::string notes = "") {
  Stats s;
  s.notes = std::move(notes);
  if (values.empty()) return s;
  std::sort(values.begin(), values.end());
  s.min = values.front();
  s.max = values.back();
  s.median = values[values.size() / 2];
  return s;
}

static void print_stat(const char *name, const Stats &s, bool last) {
  printf("    \"%s\": {\"min_gbs\": %.3f, \"median_gbs\": %.3f, \"max_gbs\": %.3f, \"notes\": \"%s\"}%s\n",
         name, s.min, s.median, s.max, s.notes.c_str(), last ? "" : ",");
}

static Stats run_device(size_t bytes, int iterations) {
  float *ptr = nullptr;
  CHECK_CUDA(cudaMalloc(&ptr, bytes));
  CHECK_CUDA(cudaMemset(ptr, 1, bytes));
  Stats s = summarize(time_kernel(ptr, bytes, iterations), "device_resident");
  CHECK_CUDA(cudaFree(ptr));
  return s;
}

static Stats run_registered(int host_node, size_t bytes, int iterations) {
  void *host = alloc_on_node(bytes, host_node);
  if (!host) return summarize({}, "numa_alloc_failed");
  touch_cpu(static_cast<float *>(host), bytes, host_node);
  int before = verify_pages(host, bytes, host_node);
  cudaError_t err = cudaHostRegister(host, bytes, cudaHostRegisterMapped);
  if (err != cudaSuccess) {
    munmap(host, bytes);
    return summarize({}, std::string("cudaHostRegister_failed:") + cudaGetErrorString(err));
  }
  int after = verify_pages(host, bytes, host_node);
  float *dev = nullptr;
  CHECK_CUDA(cudaHostGetDevicePointer(&dev, host, 0));
  std::string notes = "pages_before=" + std::to_string(before) + ";pages_after=" + std::to_string(after);
  Stats s = summarize(time_kernel(dev, bytes, iterations), notes);
  CHECK_CUDA(cudaHostUnregister(host));
  munmap(host, bytes);
  return s;
}

static Stats run_managed_cpu_first(int host_node, size_t bytes, int iterations) {
  float *ptr = nullptr;
  CHECK_CUDA(cudaMallocManaged(&ptr, bytes));
  cudaMemLocation loc{};
  loc.type = cudaMemLocationTypeHostNuma;
  loc.id = host_node;
  CHECK_CUDA(cudaMemAdvise(ptr, bytes, cudaMemAdviseSetPreferredLocation, loc));
  touch_cpu(ptr, bytes, host_node);
  int observed = verify_pages(ptr, bytes, host_node);
  Stats s = summarize(time_kernel(ptr, bytes, iterations), "cpu_preferred;observed_pages=" + std::to_string(observed));
  CHECK_CUDA(cudaFree(ptr));
  return s;
}

static Stats run_managed_gpu_first(size_t bytes, int iterations) {
  float *ptr = nullptr;
  CHECK_CUDA(cudaMallocManaged(&ptr, bytes));
  int dev = 0;
  CHECK_CUDA(cudaGetDevice(&dev));
  cudaMemLocation loc{};
  loc.type = cudaMemLocationTypeDevice;
  loc.id = dev;
  CHECK_CUDA(cudaMemAdvise(ptr, bytes, cudaMemAdviseSetPreferredLocation, loc));
  CHECK_CUDA(cudaMemset(ptr, 1, bytes));
  Stats s = summarize(time_kernel(ptr, bytes, iterations), "gpu_preferred");
  CHECK_CUDA(cudaFree(ptr));
  return s;
}

static Stats run_managed_alternating(int host_node, size_t bytes, int iterations) {
  float *ptr = nullptr;
  CHECK_CUDA(cudaMallocManaged(&ptr, bytes));
  cudaMemLocation loc{};
  loc.type = cudaMemLocationTypeHostNuma;
  loc.id = host_node;
  CHECK_CUDA(cudaMemAdvise(ptr, bytes, cudaMemAdviseSetPreferredLocation, loc));
  touch_cpu(ptr, bytes, host_node);
  int observed = verify_pages(ptr, bytes, host_node);
  Stats s = summarize(time_kernel_alternating_cpu_gpu(ptr, bytes, iterations, host_node),
                      "cpu_gpu_alternating;observed_pages=" + std::to_string(observed));
  CHECK_CUDA(cudaFree(ptr));
  return s;
}

static Stats run_unregistered_onnode(int host_node, size_t bytes, int iterations) {
  int dev = 0;
  int pageable = 0;
  CHECK_CUDA(cudaGetDevice(&dev));
  cudaError_t attr_err = cudaDeviceGetAttribute(&pageable, cudaDevAttrPageableMemoryAccess, dev);
  if (attr_err != cudaSuccess || pageable == 0) {
    return summarize({}, "skipped_pageable_memory_access_not_supported");
  }
  void *host = alloc_on_node(bytes, host_node);
  if (!host) return summarize({}, "alloc_on_node_failed");
  touch_cpu(static_cast<float *>(host), bytes, host_node);
  int observed = verify_pages(host, bytes, host_node);
  std::vector<float> values;
  std::string error;
  bool ok = time_kernel_safe(static_cast<float *>(host), bytes, iterations, &values, &error);
  munmap(host, bytes);
  if (!ok) {
    return summarize({}, "kernel_failed:" + error);
  }
  return summarize(values, "unregistered_system_memory;observed_pages=" + std::to_string(observed));
}

static Stats run_cuda_malloc_host_diag(size_t bytes, int iterations) {
  float *host = nullptr;
  cudaError_t err = cudaHostAlloc(&host, bytes, cudaHostAllocMapped);
  if (err != cudaSuccess) return summarize({}, std::string("cudaHostAlloc_failed:") + cudaGetErrorString(err));
  touch_cpu(host, bytes, 0);
  float *dev = nullptr;
  CHECK_CUDA(cudaHostGetDevicePointer(&dev, host, 0));
  Stats s = summarize(time_kernel(dev, bytes, iterations), "placement_unverified");
  CHECK_CUDA(cudaFreeHost(host));
  return s;
}

int main(int argc, char **argv) {
  if (argc != 5) {
    fprintf(stderr, "Usage: %s <gpu_id> <host_node> <buffer_bytes> <iterations>\n", argv[0]);
    return 2;
  }
  int gpu = std::atoi(argv[1]);
  int host_node = std::atoi(argv[2]);
  size_t bytes = std::strtoull(argv[3], nullptr, 10);
  int iterations = std::atoi(argv[4]);

  CHECK_CUDA(cudaSetDeviceFlags(cudaDeviceMapHost));
  CHECK_CUDA(cudaSetDevice(gpu));

  Stats device = run_device(bytes, iterations);
  Stats registered = run_registered(host_node, bytes, iterations);
  Stats host_diag = run_cuda_malloc_host_diag(bytes, iterations);
  Stats managed_cpu = run_managed_cpu_first(host_node, bytes, iterations);
  Stats managed_gpu = run_managed_gpu_first(bytes, iterations);
  Stats managed_alt = run_managed_alternating(host_node, bytes, iterations);
  Stats unregistered = run_unregistered_onnode(host_node, bytes, iterations);

  printf("{\n");
  printf("  \"config\": {\"gpu_id\": %d, \"host_node\": %d, \"buffer_bytes\": %zu, \"iterations\": %d},\n",
         gpu, host_node, bytes, iterations);
  printf("  \"modes\": {\n");
  print_stat("cudaMalloc", device, false);
  print_stat("cudaHostRegister_onnode", registered, false);
  print_stat("cudaMallocHost_diagnostic", host_diag, false);
  print_stat("cudaMallocManaged_cpu_first", managed_cpu, false);
  print_stat("cudaMallocManaged_gpu_first", managed_gpu, false);
  print_stat("cudaMallocManaged_alternating", managed_alt, false);
  print_stat("numa_alloc_onnode_unregistered_cpu_first", unregistered, true);
  printf("  }\n");
  printf("}\n");
  return 0;
}
