#include <pcap.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>

struct BufCtx {
    void* buffer;         // 存储抓包数据的缓冲区
    size_t buffer_size;   // 缓冲区大小
    size_t data_size;     // 已存储数据的大小
};

//回调函数,在 pcap_loop 捕获到符合过滤的网络包后调用
void packet_handler(u_char *user, const struct pcap_pkthdr *h, const u_char *bytes) {
    struct BufCtx *ctx = (struct BufCtx *)user;
    if (!ctx || !ctx->buffer) return;

    // 判断剩余缓冲区是否足够存放本次数据
    if (ctx->data_size + h->caplen > ctx->buffer_size) {
        fprintf(stderr, "Buffer已满\n");
        return;
    }

    // 拷贝包数据到缓冲区后续位置
    memcpy((u_char*)ctx->buffer + ctx->data_size, bytes, h->caplen);
    ctx->data_size += h->caplen;
    printf("Captured packet length: %u bytes\n", h->caplen);
}

/**
 * @brief 使用自定义过滤规则对网络数据进行抓包
 * @param iface 需要进行抓包的网络接口名称
 * @param custom_filter 用户自定义的过滤表达式
 * @param buffer 存储抓取到的数据缓冲区
 * @param buffer_size 存储抓包数据的缓冲区大小
 * @return 成功时返回0，失败返回非0错误码
 */
int custom_tcpdump_capture(const char* iface, const char* custom_filter, void* buffer, size_t buffer_size) {
    pcap_t *handle;
    char errbuf[PCAP_ERRBUF_SIZE];

    //打开网络设备
    handle = pcap_open_live(iface, 65535, 1, 1000, errbuf);
    struct bpf_program fp;
    //编译自定义的过滤规则
    pcap_compile(handle, &fp, custom_filter, 0, PCAP_NETMASK_UNKNOWN);
    //应用过滤规则
    pcap_setfilter(handle, &fp);
      
    //构造上下文，传给回调函数
    struct BufCtx ctx = {
        .buffer = buffer,
        .buffer_size = buffer_size,
        .data_size = 0
    };

    //开始进行抓包
    if (pcap_loop(handle, 0, packet_handler, (u_char*)&ctx) < 0) {
        fprintf(stderr, "pcap_loop失败: %s\n", pcap_geterr(handle));
        pcap_freecode(&fp);
        pcap_close(handle);
        return -1;
    }

    pcap_freecode(&fp);
    pcap_close(handle);

    return 0;
}

int main(int argc, char *argv[]) {
    const char* iface = argv[1];//网络接口名
    const char* filter = argv[2];//过滤表达式

    size_t buf_size = 10 * 1024 * 1024;
    void* buffer = malloc(buf_size);

    int ret = custom_tcpdump_capture(iface, filter, buffer, buf_size);

    if (ret == 0) {
        printf("Capture finished successfully.\n");
    } else {
        printf("Capture failed.\n");
    }

    free(buffer);
    return ret;
}
