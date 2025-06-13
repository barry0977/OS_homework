#include <linux/kernel.h>
#include <linux/syscalls.h>
#include <linux/slab.h>
#include <linux/sched.h>
#include <linux/fs.h>
#include <linux/errno.h>
#include <linux/uaccess.h>
#include "ramfs_file.h"

//与用户态接口连接的系统调用，参数是文件描述符fd
SYSCALL_DEFINE1(ramfs_flush_fd, int, fd)
{
    struct file *file;
    int ret;
    file = fget(fd);  // 从fd获取struct file*
    if (!file)
        return -EBADF;
    ret = ramfs_file_flush(file);
    fput(file);  // 释放引用
    return ret;
}