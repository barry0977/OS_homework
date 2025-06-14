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

// 回调函数，在 pcap_loop 捕获到符合过滤的网络包后调用
void packet_handler(u_char *user, const struct pcap_pkthdr *h, const u_char *bytes) {
    struct BufCtx *ctx = (struct BufCtx *)user;
    if (!ctx || !ctx->buffer) return;

    // 判断剩余缓冲区是否足够存放本次数据
    if (ctx->data_size + h->caplen > ctx->buffer_size) {
        // 缓冲区已满或不足，丢弃该包或可以设置标志停止抓包
        fprintf(stderr, "Buffer full, dropping packet\n");
        return;
    }

    // 拷贝包数据到缓冲区后续位置
    memcpy((u_char*)ctx->buffer + ctx->data_size, bytes, h->caplen);
    ctx->data_size += h->caplen;

    // 可选：打印抓到包的长度
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

    handle = pcap_open_live(iface, 65535, 1, 1000, errbuf);
    if (handle == NULL) {
        fprintf(stderr, "pcap_open_live failed: %s\n", errbuf);
        return -1;
    }

    struct bpf_program fp;
    if (pcap_compile(handle, &fp, custom_filter, 0, PCAP_NETMASK_UNKNOWN) == -1) {
        fprintf(stderr, "pcap_compile failed: %s\n", pcap_geterr(handle));
        pcap_close(handle);
        return -1;
    }

    if (pcap_setfilter(handle, &fp) == -1) {
        fprintf(stderr, "pcap_setfilter failed: %s\n", pcap_geterr(handle));
        pcap_freecode(&fp);
        pcap_close(handle);
        return -1;
    }

    // 构造上下文，传给回调函数
    struct BufCtx ctx = {
        .buffer = buffer,
        .buffer_size = buffer_size,
        .data_size = 0
    };

    // 捕获0表示一直抓，直到错误或中断
    if (pcap_loop(handle, 0, packet_handler, (u_char*)&ctx) < 0) {
        fprintf(stderr, "pcap_loop failed: %s\n", pcap_geterr(handle));
        pcap_freecode(&fp);
        pcap_close(handle);
        return -1;
    }

    pcap_freecode(&fp);
    pcap_close(handle);

    printf("Total captured data size: %zu bytes\n", ctx.data_size);

    return 0;
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <network_interface> <filter_expression>\n", argv[0]);
        return 1;
    }

    const char* iface = argv[1];
    const char* filter = argv[2];

    // 申请一个缓冲区存包数据（例如10MB）
    size_t buf_size = 10 * 1024 * 1024;
    void* buffer = malloc(buf_size);
    if (!buffer) {
        fprintf(stderr, "Failed to allocate memory\n");
        return 1;
    }

    int ret = custom_tcpdump_capture(iface, filter, buffer, buf_size);

    if (ret == 0) {
        printf("Capture finished successfully.\n");
        // 这里可以对 buffer 中的数据进行处理
    } else {
        printf("Capture failed.\n");
    }

    free(buffer);
    return ret;
}
