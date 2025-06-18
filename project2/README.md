# 修改部分
/home/czx/kvm/linux-5.15.178/include/linux/sched.h
/home/czx/kvm/linux-5.15.178/kernel/kv_syscalls.c
/home/czx/kvm/linux-5.15.178/arch/x86/entry/syscalls/syscall_64.tbl
/home/czx/kvm/linux-5.15.178/kernel/fork.c
/home/czx/kvm/linux-5.15.178/kernel/exit.c

# 总流程
### 修改内核后先编译内核
` make -j$(nproc) `

### 编译测试程序
`gcc -static -o kv_test kv_test.c  `

### 放入busybox
` cp project2/kv_test busybox-1.35.0/_install/bin `

### 打包
`find . -print0 \
  | cpio --null -ov --format=newc \
  | gzip -9 > ../busybox-1.35.0/initramfs.cpio.gz
`
### 进入QEMU
` qemu-system-x86_64 -kernel ./linux-5.15.178/arch/x86/boot/bzImage -initrd ./busybox-1.35.0/initramfs.cpio.gz -nographic -append "init=/init console=ttyS0" `

### 测试
`bin\kv_test `

# test 1
进入qemu后直接可以运行test1-serial

# test 2
编译为共享库，生成kvlib.so
`gcc -Wall -fPIC -shared -o kvlib.so kvlib.c -llua5.3`

在lua脚本中加载
`local kv = require("kvlib")`

进入qemu，运行sysbench
`sysbench --threads=100 --events=1000000 test2.lua run`


# test 3
make -f Makefile.test3 kbuild

将testmodule.ko移入_install/bin

启动qemu

cd bin

insmod testmodule.ko

dmesg | tail

rmmod testmodule.ko