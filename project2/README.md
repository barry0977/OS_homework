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
