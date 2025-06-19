SHELL := /bin/bash
NPROC := $(shell nproc)
KERNEL_DIR := linux-5.15.178
PROJECT_DIR := project3
BUSYBOX_DIR := busybox-1.35.0
INITRAMFS := $(BUSYBOX_DIR)/initramfs.cpio.gz
TEST_OBJECT := ramfs
RAMFS_TEST := $(PROJECT_DIR)/${TEST_OBJECT}
KERNEL_IMAGE := $(KERNEL_DIR)/arch/x86/boot/bzImage

.PHONY: all build_kernel build_test build_initramfs run clean

all: build_kernel build_test build_initramfs run

build_kernel:
	cd $(KERNEL_DIR) && make -j$(NPROC)

build_test:
	cd $(PROJECT_DIR) && gcc -static -o ${TEST_OBJECT} test1.c

build_initramfs:
	cp $(RAMFS_TEST) $(BUSYBOX_DIR)/_install/bin/
	cd $(BUSYBOX_DIR)/_install && find . -print0 | cpio --null -ov --format=newc | gzip -9 > ../initramfs.cpio.gz

build_edk:
	cd edk2/ && source edksetup.sh && build -p ProjectPkg/ProjectPkg.dsc

build:build_kernel build_initramfs build_edk

run:
	qemu-system-x86_64 -kernel $(KERNEL_IMAGE) -initrd $(INITRAMFS) -m 512M -nographic -append "init=/init console=ttyS0"

clean:
	cd $(KERNEL_DIR) && make clean
	cd $(PROJECT_DIR) && rm -f ramfs_test
	rm -f $(INITRAMFS)

