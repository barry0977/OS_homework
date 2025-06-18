# 修改文件
- linux-5.15.178
  - fs
    - ramfs
      - ramfs_file.h 
      - ramfs_persist.c
      - file-mmu.c
      - sys_flush.c

# 执行方式
cd linux-5.15.178/
make -j$(nproc)
cd ../project3/
gcc -static -o ramfs_test test1.c
cd ..
cp project3/ramfs_test busybox-1.35.0/_install/bin/
cd busybox-1.35.0/_install/
find . -print0 | cpio --null -ov --format=newc | gzip -9 > ../initramfs.cpio.gz
cd ../..
qemu-system-x86_64 -kernel ./linux-5.15.178/arch/x86/boot/bzImage -initrd ./busybox-1.35.0/initramfs.cpio.gz -nographic -append "init=/init console=ttyS0"

# 
- 已在init中挂载ramfs，并通过proc调用bind绑定持久化目录
- 进入qemu后，要测试test1，则运行ramfs1，要测试test2，则运行ramfs2，测试test3，则运行ramfs3
- 可用umount /mnt/ramfs_sync 来卸载ramfs

#
- 检测test1 flush之后文件内容是否保持一致，可用如下指令
  
cmp /mnt/ramfs_sync/testfile_single.dat /tmp/ramfs_sync/testfile_single.dat 
if [ $? -eq 0 ]; then
    echo "Files are the same"
else
    echo "Files differ"
fi

- 检测test2，先查看所有线程tid
cat /mnt/ramfs_sync/multitest.dat | grep '\[TID-' | cut -d']' -f1 | sort | uniq 

再检查所有线程是否都写入1000次
grep "\[TID-" /tmp/ramfs_sync/multitest.dat | cut -d] -f1 | sort | uniq -c 

再检查每个线程写的是否单调增长
grep "\[TID-139986408339136\]" /tmp/ramfs_sync/multitest.dat | awk '{print $NF}' | uniq -c 

还可以简单观察一下执行顺序
grep "\[TID-" /tmp/ramfs_sync/multitest.dat | head -n 100 