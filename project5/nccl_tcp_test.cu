#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <arpa/inet.h>

#define NUM_GPUS 2
#define COUNT (200 * 1024 * 1024)
#define PORT 8888

#define CUDACHECK(cmd) do { \
  cudaError_t e = cmd; \
  if (e != cudaSuccess) { \
    printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    exit(EXIT_FAILURE); \
  } \
} while (0)

#define NCCLCHECK(cmd) do { \
  ncclResult_t r = cmd; \
  if (r != ncclSuccess) { \
    printf("NCCL error %s:%d: %s\n", __FILE__, __LINE__, ncclGetErrorString(r)); \
    exit(EXIT_FAILURE); \
  } \
} while (0)

double current_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}

void run_nccl_broadcast() {
  printf("=== NCCL Broadcast 测试 ===\n");

  ncclComm_t comms[NUM_GPUS];
  float* d_buffers[NUM_GPUS];
  cudaStream_t streams[NUM_GPUS];
  int devs[NUM_GPUS] = {0, 1};

  float* h_data = (float*)malloc(COUNT * sizeof(float));
  for (int i = 0; i < COUNT; ++i) h_data[i] = float(i);

  // 每个 GPU 设置
  for (int i = 0; i < NUM_GPUS; ++i) {
    CUDACHECK(cudaSetDevice(devs[i]));
    CUDACHECK(cudaMalloc(&d_buffers[i], COUNT * sizeof(float)));
    CUDACHECK(cudaStreamCreate(&streams[i]));
  }

  // 初始化通信器
  NCCLCHECK(ncclCommInitAll(comms, NUM_GPUS, devs));

  // 主GPU拷贝数据
  CUDACHECK(cudaSetDevice(0));
  CUDACHECK(cudaMemcpy(d_buffers[0], h_data, COUNT * sizeof(float), cudaMemcpyHostToDevice));

  // 广播
  double start = current_time();
  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < NUM_GPUS; ++i)
    NCCLCHECK(ncclBroadcast(d_buffers[0], d_buffers[i], COUNT, ncclFloat, 0, comms[i], streams[i]));
  NCCLCHECK(ncclGroupEnd());

  // 等待完成
  for (int i = 0; i < NUM_GPUS; ++i) {
    CUDACHECK(cudaSetDevice(devs[i]));
    CUDACHECK(cudaStreamSynchronize(streams[i]));
  }

  double end = current_time();
  printf("NCCL Broadcast 完成，耗时 %.3f 秒，带宽 = %.2f GB/s\n",
         end - start, (COUNT * sizeof(float)) / (end - start) / 1e9);

  // 正确性检查
  float* h_recv = (float*)malloc(COUNT * sizeof(float));
  for (int i = 0; i < NUM_GPUS; ++i) {
    CUDACHECK(cudaSetDevice(devs[i]));
    CUDACHECK(cudaMemcpy(h_recv, d_buffers[i], COUNT * sizeof(float), cudaMemcpyDeviceToHost));
    for (int j = 0; j < COUNT; ++j) {
      if (h_recv[j] != float(j)) {
        printf("错误：GPU %d 数据不一致\n", i);
        break;
      }
    }
  }

  for (int i = 0; i < NUM_GPUS; ++i) {
    CUDACHECK(cudaSetDevice(devs[i]));
    CUDACHECK(cudaFree(d_buffers[i]));
    CUDACHECK(cudaStreamDestroy(streams[i]));
    ncclCommDestroy(comms[i]);
  }

  free(h_data);
  free(h_recv);
  printf("NCCL 正确性验证完成。\n");
}

void run_tcp_transfer() {
  printf("=== TCP/IP 模拟传输 ===\n");

  int sock_pair[2];
  if (socketpair(AF_UNIX, SOCK_STREAM, 0, sock_pair) != 0) {
    perror("socketpair");
    exit(EXIT_FAILURE);
  }

  pid_t pid = fork();
  if (pid == 0) {
    // 子进程模拟 Server 接收数据
    float* buffer = (float*)malloc(COUNT * sizeof(float));
    size_t total = 0;
    while (total < COUNT * sizeof(float)) {
      ssize_t n = read(sock_pair[1], ((char*)buffer) + total, COUNT * sizeof(float) - total);
      if (n < 0) {
        perror("read");
        break;
      }
      total += n;
    }
    printf("TCP 接收完成，总字节 %lu\n", total);
    free(buffer);
    exit(0);
  } else {
    // 父进程发送数据
    float* buffer = (float*)malloc(COUNT * sizeof(float));
    for (int i = 0; i < COUNT; ++i) buffer[i] = float(i);

    double start = current_time();
    write(sock_pair[0], buffer, COUNT * sizeof(float));
    double end = current_time();

    printf("TCP 发送完成，耗时 %.3f 秒，带宽 = %.2f GB/s\n",
           end - start, (COUNT * sizeof(float)) / (end - start) / 1e9);
    wait(NULL);
    free(buffer);
  }
}

int main() {
  run_nccl_broadcast();
  run_tcp_transfer();
  return 0;
}
