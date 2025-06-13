/* file-mmu.c: ramfs MMU-based file operations
 *
 * Resizable simple ram filesystem for Linux.
 *
 * Copyright (C) 2000 Linus Torvalds.
 *               2000 Transmeta Corp.
 *
 * Usage limits added by David Gibson, Linuxcare Australia.
 * This file is released under the GPL.
 */

/*
 * NOTE! This filesystem is probably most useful
 * not as a real filesystem, but as an example of
 * how virtual filesystems can be written.
 *
 * It doesn't get much simpler than this. Consider
 * that this file implements the full semantics of
 * a POSIX-compliant read-write filesystem.
 *
 * Note in particular how the filesystem does not
 * need to implement any data structures of its own
 * to keep track of the virtual data: using the VFS
 * caches is sufficient.
 */

#include <linux/fs.h>
#include <linux/mm.h>
#include <linux/ramfs.h>
#include <linux/sched.h>

#include "internal.h"

#include <linux/mutex.h>
#include <linux/slab.h>
#include <linux/path.h>
#include <linux/uaccess.h>
#include <linux/namei.h>

#include "ramfs_file.h"

#define TMP_PATH_MAX 256

static unsigned long ramfs_mmu_get_unmapped_area(struct file *file,
		unsigned long addr, unsigned long len, unsigned long pgoff,
		unsigned long flags)
{
	return current->mm->get_unmapped_area(file, addr, len, pgoff, flags);
}

/*
 * ramfs_file_flush() - 将指定 file 的内容持久化到同步目录
 * @file: 目标文件
 *
 * 1) 读取 file->f_inode 的全部内容到内核缓冲区
 * 2) 写入到 <sync_dir>/<ino>-<pid>.tmp
 * 3) 原子地 rename 到 <sync_dir>/<filename>
 */
int ramfs_file_flush(struct file *file)// 同步触发接口
{
    struct inode    *inode = file->f_inode;
    const char      *name  = file->f_path.dentry->d_name.name;
    char             tmp_path[TMP_PATH_MAX], final_path[TMP_PATH_MAX];
    struct file     *dst = NULL;
    loff_t           pos = 0;
    ssize_t          nr;
    int              ret = 0;
    void            *buf;

    struct path old_path, new_path;
    struct renamedata data;
    struct dentry *new_dentry;

    /* 检查是否已绑定 */
    if (!ramfs_sync_dir){
        pr_info("ramfs_flush: sync dir not bound\n");
        return -EINVAL;
    }

    /* 分配一个页面大小的缓冲区 */
    buf = kmalloc(PAGE_SIZE, GFP_KERNEL);
    if (!buf){
        pr_err("ramfs_flush: kmalloc failed\n");
        return -ENOMEM;
    }

    /* 构造临时文件路径 */
    snprintf(tmp_path, sizeof(tmp_path),
             "%s/%lu-%d.tmp",
             ramfs_sync_dir,
             inode->i_ino,
             task_pid_nr(current));

    /* 构造最终文件路径 */
    snprintf(final_path, sizeof(final_path),
             "%s/%s",
             ramfs_sync_dir,
             name);

    /* 打开/创建临时文件 */
    dst = filp_open(tmp_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (IS_ERR(dst)) {
        ret = PTR_ERR(dst);
        pr_err("ramfs_flush: filp_open tmp_path failed: %d\n", ret);
        goto out_free;
    }

    /* 逐页读取并写入临时文件 */
    while ((nr = kernel_read(file, buf, PAGE_SIZE, &pos)) > 0) {
        loff_t wpos = pos - nr;
        ret = kernel_write(dst, buf, nr, &wpos);
        if (ret < 0){
            pr_err("ramfs_flush: kernel_write failed: %d\n", ret);
            break;
        }
    }
    filp_close(dst, NULL);

    /* 如果写入成功，原子重命名 */
    if (ret >= 0) {
        // 先获取 old_path 和 new_path
        ret = kern_path(tmp_path, LOOKUP_FOLLOW, &old_path);
        if (ret) {
            pr_err("ramfs_flush: kern_path old_path failed: %d\n", ret);
            goto out_free;
        }

        ret = kern_path(ramfs_sync_dir, LOOKUP_FOLLOW, &new_path);
        if (ret) {
            pr_err("ramfs_flush: kern_path sync_dir failed: %d\n", ret);
            path_put(&old_path);
            goto out_free;
        }

        new_dentry = lookup_one_len(name, new_path.dentry, strlen(name));
        if (IS_ERR(new_dentry)) {
            ret = PTR_ERR(new_dentry);
            pr_err("ramfs_flush: lookup_one_len failed: %d\n", ret);
            path_put(&old_path);
            path_put(&new_path);
            goto out_free;
        }

        memset(&data, 0, sizeof(data));
        data.old_dir = old_path.dentry->d_parent->d_inode;
        data.old_dentry = old_path.dentry;

        data.new_dir = new_path.dentry->d_inode;
        data.new_dentry = new_dentry;
        data.flags = 0;

        ret = vfs_rename(&data);
        if (ret){
            pr_err("ramfs_flush: vfs_rename failed: %d\n", ret);
        }
        
        dput(new_dentry);
        path_put(&old_path);
        path_put(&new_path);
    }

    if (ret < 0)
        pr_err("ramfs: flush '%s' failed: %d\n", name, ret);

out_free:
    kfree(buf);
    return ret;
}
EXPORT_SYMBOL(ramfs_file_flush);


const struct file_operations ramfs_file_operations = {
	.read_iter	= generic_file_read_iter,
	.write_iter	= generic_file_write_iter,
	.mmap		= generic_file_mmap,
	.fsync		= noop_fsync, 
	.splice_read	= generic_file_splice_read,
	.splice_write	= iter_file_splice_write,
	.llseek		= generic_file_llseek,
	.get_unmapped_area	= ramfs_mmu_get_unmapped_area,
};

const struct inode_operations ramfs_file_inode_operations = {
	.setattr	= simple_setattr,
	.getattr	= simple_getattr,
};
