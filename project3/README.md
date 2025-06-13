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
- 已在init中挂载
- 进入qemu后，要测试test1，则运行ramfs，要测试test2，则运行checkramfs2
### 进入qemu后
