#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/proc_fs.h>
#include <linux/uaccess.h>
#include <linux/fs.h>
#include <linux/mutex.h>
#include <linux/slab.h>
#include <linux/namei.h>

#define PROC_NAME "ramfs_bind"

//新增的全局变量
char *ramfs_sync_dir;
struct path ramfs_sync_path;
static struct mutex ramfs_sync_mutex;

/*
 * ramfs_bind() - 绑定一个真实目录，用于存放持久化文件
 * @sync_dir: 用户态传入的目录字符串
 *
 * 验证目录是否存在、是否为目录、是否可写，然后将其路径复制保存，
 * 并解析成 struct path 供后续 flush 使用。
 */
int ramfs_bind(const char *sync_dir)
{
    struct path path;
    char *new_dir;
    int ret;

    if (!sync_dir)
        return -EINVAL;

    /* 验证目录存在且为目录 */
    ret = kern_path(sync_dir, LOOKUP_DIRECTORY, &path);
    if (ret)
        return ret;

    /* 检查写权限 */
    if (!(d_inode(path.dentry)->i_mode & S_IWUSR)) {
        path_put(&path);
        return -EACCES;
    }

    /* 复制路径字符串到内核 */
    new_dir = kstrdup(sync_dir, GFP_KERNEL);
    if (!new_dir) {
        path_put(&path);
        return -ENOMEM;
    }

    mutex_lock(&ramfs_sync_mutex);
    /* 释放旧的目录字符串 */
    kfree(ramfs_sync_dir);
    ramfs_sync_dir = new_dir;

    /* 保存解析好的 path，用于后续重命名 */
    path_put(&ramfs_sync_path);
    ramfs_sync_path = path;
    mutex_unlock(&ramfs_sync_mutex);

    pr_info("ramfs: bound to sync dir '%s'\n", ramfs_sync_dir);
    return 0;
}

// proc写操作
static ssize_t ramfs_bind_write(struct file *file, const char __user *buf, size_t count, loff_t *pos)
{
    char *sync_dir;
    ssize_t ret;

    sync_dir = kzalloc(count + 1, GFP_KERNEL);
    if (!sync_dir)
        return -ENOMEM;

    if (copy_from_user(sync_dir, buf, count)) {
        ret = -EFAULT;
        goto out;
    }

    sync_dir[count] = '\0'; // 确保路径字符串以 '\0' 结尾

    ret = ramfs_bind(sync_dir);

out:
    kfree(sync_dir);
    return ret ? ret : count;
}

// proc 文件操作结构
static const struct proc_ops ramfs_bind_fops = {
    .proc_write = ramfs_bind_write,
};

// 模块初始化
static int __init ramfs_bind_init(void)
{
    struct proc_dir_entry *entry;

    // 创建 /proc/ramfs_bind 文件
    entry = proc_create(PROC_NAME, 0666, NULL, &ramfs_bind_fops);
    if (!entry) {
        pr_err("Failed to create /proc/%s\n", PROC_NAME);
        return -ENOMEM;
    }

    pr_info("RAMfs bind module loaded\n");
    return 0;
}

// 模块清除
static void __exit ramfs_bind_exit(void)
{
    if (ramfs_sync_dir) {
        kfree(ramfs_sync_dir);
        pr_info("RAMfs sync directory unbound\n");
    }

    remove_proc_entry(PROC_NAME, NULL);
    pr_info("RAMfs bind module unloaded\n");
}

module_init(ramfs_bind_init);
module_exit(ramfs_bind_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Your Name");
MODULE_DESCRIPTION("RAMfs Bind to Sync Directory");
