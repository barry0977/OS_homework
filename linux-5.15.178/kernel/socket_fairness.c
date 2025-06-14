#include <linux/kernel.h>
#include <linux/syscalls.h>
#include <linux/sched/signal.h>
#include <linux/uaccess.h>
#include <linux/cred.h>

/*
* @brief 为特定线程配置Socket级别的公平管理策略
* @param thread_id
需要配置的线程ID
* @param max_socket_allowed 该线程允许打开的最大Socket数
* @param priority_level 该线程的优先级（影响Socket分配策略）
* @return 成功时返回0，失败返回非0错误码
*/
SYSCALL_DEFINE3(configure_socket_fairness, pid_t, pid, int, max_socket, int, priority)
{
    struct task_struct *task;

    rcu_read_lock();
    task = find_task_by_vpid(pid);
    if (!task) {
        rcu_read_unlock();
        return -ESRCH; // No such process
    }

    // 权限检查：必须是root或者自己
    if (!uid_eq(current_uid(), task->real_cred->uid) && !capable(CAP_SYS_ADMIN)) {
        rcu_read_unlock();
        return -EPERM;
    }

    // 检查最大Socket数的有效性
    if (max_socket < 0 || max_socket > 1024) { 
        rcu_read_unlock();
        return -EINVAL;
    }

    task->socket_max_allowed = max_socket;
    task->socket_priority = priority;

    rcu_read_unlock();
    return 0;
}
