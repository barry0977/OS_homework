#define _GNU_SOURCE
#include <stdio.h>
#include <unistd.h>
#include <sys/syscall.h>

#define SYS_ramfs_flush_fd 451

void ramfs_file_flush(FILE *file) {
    int fd = fileno(file);  // 从 FILE* 获取 fd
    if (fd < 0) {
        perror("Invalid file pointer");
        return;
    }
    int ret = syscall(SYS_ramfs_flush_fd, fd);
    if (ret < 0)
        perror("ramfs_file_flush syscall failed");
}