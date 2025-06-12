#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/xattr.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#define SET_XATTR 188
#define READ_KV_SYSCALL 191
#define REMOVE_XATTR 197


int set_xattr(const char *path, const char *name, const char *value){
    size_t size = strlen(value);//需要设置属性值的大小，否则后面无法正确添加
    int flags = 0; //默认不设置任何标志
    int result = syscall(SET_XATTR, path, name, value,size,flags);//0-成功,-1-失败
    return result + 1;
};

// Function to get an extended attribute
// If name does not exist, return -1
// If name exists, return 1 and copy value to dst
char* get_xattr(const char *path, const char *name){
    ssize_t size = syscall(READ_KV_SYSCALL,path,name,NULL,0); //先获取属性值的字节数
    char *value = (char *)malloc(size+1); //留出一个字节空间用于字符串结束符
    ssize_t result = syscall(READ_KV_SYSCALL, path, name, value, size); //读取属性值存储到value中
    if(result == -1) {
        free(value);
        return NULL; // 如果读取失败，返回NULL
    }else{
        value[size] = '\0';
        return value;
    }
};

// Function to remove an extended attribute
int remove_xattr(const char *path, const char *name){
    int result =  syscall(REMOVE_XATTR, path, name);//0-成功,-1-失败
    return result + 1; 
};

void get_inode_info(const char *path) {
    struct stat st;
    if (stat(path, &st) == -1) {//获取文件信息
        perror("stat");
        return;
    }
    
    printf("Inode info for %s:\n", path);
    printf("  Inode number: %lu\n", st.st_ino);
    printf("  File type: ");
    
    //识别文件类型
    switch (st.st_mode & S_IFMT) {
        case S_IFREG:  printf("Regular file\n"); break;//普通文件
        case S_IFDIR:  printf("Directory\n"); break;//目录
        case S_IFLNK:  printf("Symbolic link\n"); break;//符号链接
        default:       printf("Other\n"); break;
    }
}

void list_xattrs(const char *path) {
    ssize_t size = listxattr(path, NULL, 0);//
    if (size == -1) {
        perror("listxattr");
        return;
    }
    if (size == 0) {
        printf("No extended attributes found.\n");
        return;
    }
    char *list = (char*)malloc(size);
    ssize_t ret = listxattr(path, list, size);//将所有扩展属性名存储到list中,返回以\0 分隔的连续字符串
    if (ret == -1) {
        perror("listxattr");
        free(list);
        return;
    }
    int index = 0;
    char *attr = list;
    while (attr < list + ret) {
        char* value = get_xattr(path, attr);//获取对应的属性值
        printf("Extended attribute %d: %s, %s\n", index++, attr,value);
        free(value); // 释放获取的属性值
        attr += strlen(attr) + 1;  // 跳到下一个属性名
    }

    free(list);
}
