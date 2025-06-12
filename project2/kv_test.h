#include <unistd.h>
#include <sys/syscall.h>

#define SYS_write_kv 449
#define SYS_read_kv  450

int write_kv(int k, int v) {
    return syscall(SYS_write_kv, k, v);
}

int read_kv(int k) {
    return syscall(SYS_read_kv, k);
}