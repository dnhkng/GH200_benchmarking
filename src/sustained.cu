#include <cuda_runtime.h>
#include <linux/mempolicy.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

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
}

static int stream_device(const Endpoint &src, const Endpoint &dst) {
  if (dst.device) return dst.gpu;
  if (src.device) return src.gpu;
  return 0;
}

static void do_copy(const Endpoint &src, const Endpoint &dst, cudaStream_t stream) {
  if (!src.device && !dst.device) {
    memcpy(dst.ptr, src.ptr, src.bytes);
    return;
  }
  if (src.device && dst.device && src.gpu != dst.gpu) {
    CHECK_CUDA(cudaMemcpyPeerAsync(dst.ptr, dst.gpu, src.ptr, src.gpu, src.bytes, stream));
  } else {
    CHECK_CUDA(cudaMemcpyAsync(dst.ptr, src.ptr, src.bytes, cudaMemcpyDefault, stream));
  }
}

int main(int argc, char **argv) {
  if (argc != 5) {
    fprintf(stderr, "Usage: %s <src_node> <dst_node> <chunk_bytes> <num_chunks>\n", argv[0]);
    return 2;
  }
  std::string src_name = argv[1];
  std::string dst_name = argv[2];
  size_t chunk = strtoull(argv[3], nullptr, 10);
  size_t chunks = strtoull(argv[4], nullptr, 10);

  Endpoint src = parse_endpoint(src_name, chunk);
  Endpoint dst = parse_endpoint(dst_name, chunk);
  allocate_endpoint(src);
  allocate_endpoint(dst);

  int dev = stream_device(src, dst);
  CHECK_CUDA(cudaSetDevice(dev));
  cudaStream_t stream{};
  CHECK_CUDA(cudaStreamCreate(&stream));

  auto start = std::chrono::steady_clock::now();
  for (size_t i = 0; i < chunks; ++i) {
    do_copy(src, dst, stream);
  }
  CHECK_CUDA(cudaStreamSynchronize(stream));
  auto end = std::chrono::steady_clock::now();

  double seconds = std::chrono::duration<double>(end - start).count();
  double total_bytes = static_cast<double>(chunk) * static_cast<double>(chunks);
  double gbs = total_bytes / seconds / 1e9;
  printf("src=%s\n", src_name.c_str());
  printf("dst=%s\n", dst_name.c_str());
  printf("chunk_bytes=%zu\n", chunk);
  printf("num_chunks=%zu\n", chunks);
  printf("seconds=%.6f\n", seconds);
  printf("gbps=%.6f\n", gbs);

  CHECK_CUDA(cudaStreamDestroy(stream));
  free_endpoint(src);
  free_endpoint(dst);
  return 0;
}
