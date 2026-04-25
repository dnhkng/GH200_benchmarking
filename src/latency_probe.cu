#include <cuda_runtime.h>
#include <linux/mempolicy.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
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

struct Endpoint {
  bool device = false;
  int gpu = -1;
  int host_node = -1;
  void *ptr = nullptr;
  size_t bytes = 0;
};

static int env_int(const char *name, int fallback) {
  const char *value = getenv(name);
  return value ? atoi(value) : fallback;
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

static Endpoint parse_endpoint(const std::string &name, size_t bytes) {
  Endpoint ep;
  ep.bytes = bytes;
  if (name == "h0") {
    ep.device = true; ep.gpu = 0;
  } else if (name == "h1") {
    ep.device = true; ep.gpu = 1;
  } else if (name == "l0") {
    ep.device = false; ep.host_node = env_int("GPU_0_NODE", 0);
  } else if (name == "l1") {
    ep.device = false; ep.host_node = env_int("GPU_1_NODE", 1);
  } else {
    fprintf(stderr, "unknown endpoint: %s\n", name.c_str());
    exit(2);
  }
  return ep;
}

static void touch_host(void *ptr, size_t bytes, int node) {
  bind_to_node_cpus(node);
  volatile char *p = reinterpret_cast<volatile char *>(ptr);
  long ps = sysconf(_SC_PAGESIZE);
  for (size_t off = 0; off < bytes; off += ps) p[off] = static_cast<char>(off);
}

static void allocate_endpoint(Endpoint &ep) {
  if (ep.device) {
    CHECK_CUDA(cudaSetDevice(ep.gpu));
    CHECK_CUDA(cudaMalloc(&ep.ptr, ep.bytes));
    CHECK_CUDA(cudaMemset(ep.ptr, 1, ep.bytes));
  } else {
    ep.ptr = alloc_on_node(ep.bytes, ep.host_node);
    if (!ep.ptr) {
      fprintf(stderr, "alloc_on_node failed for node %d\n", ep.host_node);
      exit(2);
    }
    touch_host(ep.ptr, ep.bytes, ep.host_node);
    cudaError_t err = cudaHostRegister(ep.ptr, ep.bytes, cudaHostRegisterDefault);
    if (err != cudaSuccess) {
      fprintf(stderr, "cudaHostRegister failed: %s\n", cudaGetErrorString(err));
      exit(2);
    }
  }
}

static void free_endpoint(Endpoint &ep) {
  if (!ep.ptr) return;
  if (ep.device) {
    CHECK_CUDA(cudaSetDevice(ep.gpu));
    CHECK_CUDA(cudaFree(ep.ptr));
  } else {
    CHECK_CUDA(cudaHostUnregister(ep.ptr));
    munmap(ep.ptr, ep.bytes);
  }
  ep.ptr = nullptr;
}

static int stream_device(const Endpoint &src, const Endpoint &dst) {
  if (dst.device) return dst.gpu;
  if (src.device) return src.gpu;
  return 0;
}

static void do_copy(const Endpoint &src, const Endpoint &dst, cudaStream_t stream) {
  if (!src.device && !dst.device) {
    memcpy(dst.ptr, src.ptr, std::min(src.bytes, dst.bytes));
    return;
  }
  if (src.device && dst.device && src.gpu != dst.gpu) {
    CHECK_CUDA(cudaMemcpyPeerAsync(dst.ptr, dst.gpu, src.ptr, src.gpu, src.bytes, stream));
  } else {
    CHECK_CUDA(cudaMemcpyAsync(dst.ptr, src.ptr, src.bytes, cudaMemcpyDefault, stream));
  }
}

static double percentile(std::vector<double> values, double p) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  size_t idx = static_cast<size_t>((values.size() - 1) * p);
  return values[idx];
}

static void run_case(const std::string &mode, const std::string &src_name, const std::string &dst_name, size_t size) {
  Endpoint src = parse_endpoint(src_name, size);
  Endpoint dst = parse_endpoint(dst_name, size);
  allocate_endpoint(src);
  allocate_endpoint(dst);

  int dev = stream_device(src, dst);
  CHECK_CUDA(cudaSetDevice(dev));
  cudaStream_t stream{};
  CHECK_CUDA(cudaStreamCreate(&stream));

  int warmup = 100;
  int iterations = 1000;
  for (int i = 0; i < warmup; ++i) {
    do_copy(src, dst, stream);
    if (mode == "roundtrip_lat") do_copy(dst, src, stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));
  }

  std::vector<double> us;
  us.reserve(iterations);
  for (int i = 0; i < iterations; ++i) {
    auto start = std::chrono::steady_clock::now();
    do_copy(src, dst, stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    if (mode == "roundtrip_lat") {
      do_copy(dst, src, stream);
      CHECK_CUDA(cudaStreamSynchronize(stream));
    }
    auto end = std::chrono::steady_clock::now();
    us.push_back(std::chrono::duration<double, std::micro>(end - start).count());
  }

  double med = percentile(us, 0.50);
  double p99 = percentile(us, 0.99);
  double bytes = mode == "roundtrip_lat" ? 2.0 * size : static_cast<double>(size);
  double gbs = med > 0.0 ? bytes / (med * 1e-6) / 1e9 : 0.0;
  printf("%s,%s,%s,%zu,%d,%.3f,%.3f,%.6f\n",
         mode.c_str(), src_name.c_str(), dst_name.c_str(), size, iterations, med, p99, gbs);

  CHECK_CUDA(cudaStreamDestroy(stream));
  free_endpoint(src);
  free_endpoint(dst);
}

int main(int argc, char **argv) {
  if (argc != 4) {
    fprintf(stderr, "Usage: %s <oneway_bw|roundtrip_lat> <src> <dst>\n", argv[0]);
    return 2;
  }
  std::string mode = argv[1];
  std::string src = argv[2];
  std::string dst = argv[3];
  std::vector<size_t> sizes = {8, 64, 512, 4096, 32768, 262144, 2097152, 16777216};
  printf("mode,src,dst,size_bytes,iterations,median_us,p99_us,gbs\n");
  for (size_t size : sizes) run_case(mode, src, dst, size);
  return 0;
}
