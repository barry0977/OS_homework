#include <linux/kernel.h>
#include <linux/syscalls.h>
#include <linux/slab.h>
#include <linux/sched.h>

SYSCALL_DEFINE2(write_kv, int, k, int, v)
{
    struct task_struct *task = current->group_leader; // 获得当前线程组中最开始创建的那个线程，负责储存kv键值对
    int index = k % 1024; //采用开放寻址哈希表实现
    struct hlist_head *head = &task->kv_store[index];
    spinlock_t *lock = &task->kv_lock[index];
    struct kv_node *entry;

    spin_lock(lock);
    //先查找旧的key，如果存在就更新
    hlist_for_each_entry(entry, head, node) {
        if (entry->key == k) {
            entry->value = v;
            spin_unlock(lock);
            return sizeof(int);
        }
    }
    //若原来不存在，则新分配
    entry = kmalloc(sizeof(*entry), GFP_KERNEL);
    if (!entry) {
        spin_unlock(lock);
        return -1;
    }
    entry->key = k;
    entry->value = v;
    hlist_add_head(&entry->node, head);
    spin_unlock(lock);
    return sizeof(int);
}

SYSCALL_DEFINE1(read_kv, int, k)
{
    struct task_struct *task = current->group_leader;
    int index = k % 1024;
    struct hlist_head *head = &task->kv_store[index];
    spinlock_t *lock = &task->kv_lock[index];
    struct kv_node *entry;
    int ret = -1;

    spin_lock(lock);
    hlist_for_each_entry(entry, head, node) {
        if (entry->key == k) {
            ret = entry->value;
            break;
        }
    }
    spin_unlock(lock);

    return ret;
}