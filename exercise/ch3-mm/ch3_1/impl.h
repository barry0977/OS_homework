#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>

/**
 * @brief 重新映射一块虚拟内存区域
 * @param addr 原始映射的内存地址，如果为 NULL 则由系统自动选择一个合适的地址
 * @param size 需要映射的大小（单位：字节）
 * @return 成功返回映射的地址，失败返回 NULL
 * @details 该函数用于重新映射一个新的虚拟内存区域。如果 addr 参数为 NULL，
 *          系统会自动选择一个合适的地址进行映射。映射的内存区域大小为 size 字节。
 *          映射失败时返回 NULL。
 */
void* mmap_remap(void *addr, size_t size) {
    void* new_addr = mmap(NULL,size,PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);//不需要文件映射，fd 设置为 -1，表示匿名映射
    if (new_addr == MAP_FAILED) {
        perror("mmap_remap failed");
        return NULL;
    }
    if (addr != NULL) {
         memcpy(new_addr, addr, size);  //将原有映射的内容复制到新的映射地址
        //解除原有映射
        if (munmap(addr, size) == -1) {
            perror("munmap failed");
            return NULL; //解除映射失败，返回 NULL
        }
    }
    return new_addr; //返回新的映射地址
}

/**
 * @brief 使用 mmap 进行文件读写
 * @param filename 待操作的文件路径
 * @param offset 写入文件的偏移量（单位：字节）
 * @param content 要写入文件的内容
 * @return 成功返回 0，失败返回 -1
 * @details 该函数使用内存映射（mmap）的方式进行文件写入操作。
 *          通过 filename 指定要写入的文件，
 *          offset 指定写入的起始位置，
 *          content 指定要写入的内容。
 *          写入成功返回 0，失败返回 -1。
 */
int file_mmap_write(const char* filename, size_t offset, char* content) {
    //0644: - 文件所有者：读 + 写  - 同组用户：只读  - 其他用户：只读
    int fd = open(filename, O_RDWR | O_CREAT, 0644);
    if (fd < 0) {
        perror("open");
        return -1;
    }

    size_t content_len = strlen(content);
    size_t required_size = offset + content_len;//文件需要的最小大小

    struct stat st;
    if (fstat(fd, &st) == -1) {
        perror("fstat");
        close(fd);
        return -1;
    }

    if ((size_t)st.st_size < required_size) {//如果文件大小小于需要的大小，则需要扩展文件
        if (ftruncate(fd, required_size) == -1) {
            perror("ftruncate");
            close(fd);
            return -1;
        }
    }

    void *map = mmap(NULL, required_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);//需要用MAP_SHARED,否则修改内存内容无法影响文件
    if (map == MAP_FAILED) {
        perror("mmap");
        close(fd);
        return -1;
    }

    memcpy((char *)map + offset, content, content_len);

    //把内存映射中的修改同步到磁盘
    if (msync(map, required_size, MS_SYNC) == -1) {
        perror("msync");
    }

    munmap(map, required_size);
    close(fd);
    return 0;
}
