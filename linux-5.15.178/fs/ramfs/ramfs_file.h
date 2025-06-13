#ifndef RAMFS_FILE_H
#define RAMFS_FILE_H

#include <linux/fs.h>  // 需要 struct file 声明
#include <linux/path.h> // 需要 struct path 声明

int ramfs_file_flush(struct file *file);

extern char *ramfs_sync_dir;
extern struct path ramfs_sync_path;

#endif // RAMFS_FILE_H
