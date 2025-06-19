#include <stdio.h>
#include <stdlib.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include <math.h>   
#include <time.h>
#define NUM_GPUS 2 
#define COUNT (1024 * 1024 * 50) 


#define CUDACHECK(cmd) do {                         \
  cudaError_t err = cmd;                            \
  if (err != cudaSuccess) {                         \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(err)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define NCCLCHECK(cmd) do {                         \
  ncclResult_t res = cmd;                           \
  if (res != ncclSuccess) {                         \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(res)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char* argv[]) {
    int device_count = 0;
    CUDACHECK(cudaGetDeviceCount(&device_count));
    if (device_count < 2) {
        printf("需要至少2个GPU设备来运行测试程序，当前有%d个设备\n", device_count);
        return 0;
    }

    float* d_send[NUM_GPUS];
    float* d_recv[NUM_GPUS];
    float* h_buffer = (float*)malloc(COUNT * sizeof(float));
    ncclComm_t comms[NUM_GPUS];
    cudaStream_t streams[NUM_GPUS];

    int devices[NUM_GPUS] = {0, 1};

    //NCCL 初始化
    NCCLCHECK(ncclCommInitAll(comms, NUM_GPUS, devices));

    //为每个GPU分配缓冲
    for (int i = 0; i < NUM_GPUS; ++i) {
        CUDACHECK(cudaSetDevice(devices[i]));
        CUDACHECK(cudaMalloc((void**)&d_send[i], COUNT * sizeof(float)));
        CUDACHECK(cudaMalloc((void**)&d_recv[i], COUNT * sizeof(float)));
    }

    //初始化 root GPU 数据
    for (size_t i = 0; i < COUNT; ++i) h_buffer[i] = (float)i;
    CUDACHECK(cudaSetDevice(devices[0]));
    CUDACHECK(cudaMemcpy(d_send[0], h_buffer, COUNT * sizeof(float), cudaMemcpyHostToDevice));

    printf("===== NCCL Broadcast 测试 =====\n");
    double start = get_time();
    //每个 GPU 创建 stream
    for (int i = 0; i < NUM_GPUS; ++i) {
        CUDACHECK(cudaSetDevice(devices[i]));
        CUDACHECK(cudaStreamCreate(&streams[i]));
    }

    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < NUM_GPUS; ++i) {
        CUDACHECK(cudaSetDevice(devices[i]));
        NCCLCHECK(ncclBroadcast(
            d_send[i],
            d_send[i],
            COUNT,
            ncclFloat,
            0,
            comms[i],
            streams[i]
        ));
    }
    NCCLCHECK(ncclGroupEnd());

    //cuda间同步
    for (int i = 0; i < NUM_GPUS; ++i) {
        CUDACHECK(cudaSetDevice(devices[i]));
        CUDACHECK(cudaStreamSynchronize(streams[i]));
    }

    double end = get_time();
    double duration = end - start;
    double gb = (COUNT * sizeof(float)) / 1e9;
    printf("广播完成，耗时 %.4f 秒，带宽约 %.2f GB/s\n", duration, gb / duration);

    for (int i = 0; i < NUM_GPUS; ++i) {
        float* h_check = (float*)malloc(COUNT * sizeof(float));
        CUDACHECK(cudaMemcpy(h_check, d_send[i], COUNT * sizeof(float), cudaMemcpyDeviceToHost));
        int mismatch = 0;
        for (size_t j = 0; j < COUNT; ++j) {
            if (fabs(h_check[j] - (float)j) > 1e-5) {
                printf("GPU %d 数据 mismatch at %lu: %f != %f\n", i, j, h_check[j], (float)j);
                mismatch = 1;
                break;
            }
        }
        if (!mismatch) printf("GPU %d 数据一致 (Broadcast)\n", i);
        free(h_check);
    }

    printf("\n===== NCCL AllReduce 测试 =====\n");
    for (int i = 0; i < NUM_GPUS; ++i) {
        float val = (float)(i + 1);  //gpu 0: 1.0, gpu 1: 2.0
        CUDACHECK(cudaSetDevice(devices[i]));
        CUDACHECK(cudaMemset(d_send[i], 0, COUNT * sizeof(float)));
        CUDACHECK(cudaMemset(d_recv[i], 0, COUNT * sizeof(float)));
        float* h_init = (float*)malloc(COUNT * sizeof(float));
        for (size_t j = 0; j < COUNT; ++j) h_init[j] = val;
        CUDACHECK(cudaMemcpy(d_send[i], h_init, COUNT * sizeof(float), cudaMemcpyHostToDevice));
        free(h_init);
    }

    start = get_time();
    printf("开始 AllReduce 操作...\n");
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < NUM_GPUS; ++i) {
        CUDACHECK(cudaSetDevice(devices[i]));
        CUDACHECK(cudaSetDevice(devices[i]));
        NCCLCHECK(ncclAllReduce(
            d_send[i],
            d_recv[i],
            COUNT,
            ncclFloat,
            ncclSum,
            comms[i],
            streams[i]
        ));
    }
    NCCLCHECK(ncclGroupEnd());
    for (int i = 0; i < NUM_GPUS; ++i) {
        CUDACHECK(cudaSetDevice(devices[i]));
        CUDACHECK(cudaStreamSynchronize(streams[i]));
    }
    end = get_time();
    duration = end - start;
    printf("AllReduce 完成，耗时 %.4f 秒，带宽约 %.2f GB/s\n", duration, gb / duration);

    //正确性检查（预期值为每个元素的总和: 1.0 + 2.0）
    float expected = 0.0f;
    for (int i = 0; i < NUM_GPUS; ++i) expected += (float)(i + 1);

    for (int i = 0; i < NUM_GPUS; ++i) {
        float* h_check = (float*)malloc(COUNT * sizeof(float));
        CUDACHECK(cudaMemcpy(h_check, d_recv[i], COUNT * sizeof(float), cudaMemcpyDeviceToHost));
        int mismatch = 0;
        for (size_t j = 0; j < COUNT; ++j) {
            if (fabs(h_check[j] - expected) > 1e-4) {
                printf("AllReduce mismatch on GPU %d at %lu: %f != %f\n", i, j, h_check[j], expected);
                mismatch = 1;
                break;
            }
        }
        if (!mismatch) printf("GPU %d AllReduce 正确\n", i);
        free(h_check);
    }

    for (int i = 0; i < NUM_GPUS; ++i) {
        CUDACHECK(cudaSetDevice(devices[i]));
        CUDACHECK(cudaStreamDestroy(streams[i]));
    }

    for (int i = 0; i < NUM_GPUS; ++i) {
        CUDACHECK(cudaFree(d_send[i]));
        CUDACHECK(cudaFree(d_recv[i]));
        ncclCommDestroy(comms[i]);
    }

    free(h_buffer);
    return 0;
}