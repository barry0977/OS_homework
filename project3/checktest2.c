#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <pthread.h>
#include <time.h>
#include <sys/types.h>
#include "ramfs.h"

#define THREAD_PRODUCERS 10
#define THREAD_CONSUMERS 3
#define ITERATIONS 1000
#define FILEPATH "./multitest.dat"
#define ENTRY_SIZE 128

pthread_mutex_t write_lock;

void *producer_thread(void *arg) {
    int tid = *(int*)arg;
    int fd = open(FILEPATH, O_CREAT | O_WRONLY | O_APPEND, 0666);
    if(fd < 0) {
        perror("Producer open failed");
        pthread_exit(NULL);
    }

    char entry[ENTRY_SIZE];
    for(int i = 0; i < ITERATIONS; i++) {
        // timestamp精确到秒，可用gettimeofday提高精度
        long timestamp = time(NULL);
        snprintf(entry, ENTRY_SIZE, "[TID-%d] timestamp: %ld iteration: %d\n", tid, timestamp, i);

        pthread_mutex_lock(&write_lock);
        if(write(fd, entry, strlen(entry)) < 0) perror("Producer write failed");
        pthread_mutex_unlock(&write_lock);

        usleep(rand() % 1000); // 模拟变动负载
    }

    close(fd);
    pthread_exit(NULL);
}


void *consumer_thread(void *arg) {
    int fd = open(FILEPATH, O_CREAT | O_WRONLY | O_APPEND, 0666);
    for(int i = 0; i < ITERATIONS; i++) {
        pthread_mutex_lock(&write_lock); // 保护flush与写操作
        ramfs_file_flush(fd);
        printf("Consumer thread %ld triggered flush %d\n", pthread_self(), i);
        pthread_mutex_unlock(&write_lock);

        usleep((rand() % 2000) + 500);
    }
    pthread_exit(NULL);
}

// 验证日志的函数：简单检查每个线程迭代号单调递增
int verify_log_file(const char *filepath) {
    FILE *fp = fopen(filepath, "r");
    if (!fp) {
        perror("Open log file failed");
        return -1;
    }

    char line[256];
    int last_iter[THREAD_PRODUCERS];
    memset(last_iter, -1, sizeof(last_iter));
    int line_num = 0;
    int errors = 0;

    while (fgets(line, sizeof(line), fp)) {
        line_num++;
        int tid, iter;
        long timestamp;
        int matched = sscanf(line, "[TID-%d] timestamp: %ld iteration: %d", &tid, &timestamp, &iter);
        if (matched != 3) {
            printf("Malformed line %d: %s", line_num, line);
            errors++;
            continue;
        }
        if (tid < 0 || tid >= THREAD_PRODUCERS) {
            printf("Invalid tid at line %d: %d\n", line_num, tid);
            errors++;
            continue;
        }
        if (iter <= last_iter[tid]) {
            printf("Order violation for tid %d at line %d: %d <= %d\n", tid, line_num, iter, last_iter[tid]);
            errors++;
        }
        last_iter[tid] = iter;
    }
    fclose(fp);

    if(errors == 0) {
        printf("Log verification passed: all thread iterations are sequentially consistent.\n");
    } else {
        printf("Log verification found %d errors.\n", errors);
    }
    return errors;
}

int main() {
    pthread_t producers[THREAD_PRODUCERS], consumers[THREAD_CONSUMERS];
    int tids[THREAD_PRODUCERS];
    srand(time(NULL));
    pthread_mutex_init(&write_lock, NULL);

    // 删除旧日志
    unlink(FILEPATH);

    printf("Starting multi-threaded sequential consistency test...\n");

    for (int i = 0; i < THREAD_PRODUCERS; i++) {
        tids[i] = i;
        pthread_create(&producers[i], NULL, producer_thread, &tids[i]);
    }
    for (int i = 0; i < THREAD_CONSUMERS; i++) {
        pthread_create(&consumers[i], NULL, consumer_thread, NULL);
    }

    for (int i = 0; i < THREAD_PRODUCERS; i++) {
        pthread_join(producers[i], NULL);
    }
    for (int i = 0; i < THREAD_CONSUMERS; i++) {
        pthread_join(consumers[i], NULL);
    }

    pthread_mutex_destroy(&write_lock);

    printf("Multi-threaded test completed. Verifying logs...\n");

    verify_log_file(FILEPATH);

    return 0;
}
