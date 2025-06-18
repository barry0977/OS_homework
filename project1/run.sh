#!/bin/bash
 #获取脚本所在目录的绝对路径
#SCRIPT_DIR="$(cd "$(dirname"${BASH_SOURCE[0]}")" && pwd)"
 #获取项目根目录（假设脚本在tools/scripts目录下）
#PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
 #定义资源文件路径
BZIMAGE="/home/czx/kvm/linux-5.15.178/arch/x86/boot/bzImage"
INITRD="/home/czx/kvm/busybox-1.35.0/initramfs.cpio.gz"
OVMF_CODE="/home/czx/kvm/edk2/Build/Ovmf3264/DEBUG_GCC5/FV/OVMF_CODE.fd"
OVMF_VARS="/home/czx/kvm/edk2/Build/Ovmf3264/DEBUG_GCC5/FV/OVMF_VARS.fd"
HELLO_WORLD_EFI="/home/czx/kvm/edk2/Build/Test/DEBUG_GCC5/X64/HelloWorld.efi"
MY_ACPIVIEW_EFI="/home/czx/kvm/edk2/Build/MyACPI/DEBUG_GCC5/X64/MyACPI.efi"
Change_ACPI_EFI="/home/czx/kvm/edk2/Build/MyACPI/DEBUG_GCC5/X64/ChangeACPI.efi"
New_ACPI_EFI="/home/czx/kvm/edk2/Build/MyACPI/DEBUG_GCC5/X64/NewACPI.efi"
#检查上述文件是否存在

#RESOURCE_LIST=("$OVMF_CODE""$OVMF_VARS""$RAW_ACPIVIEW_EFI" "$HELLO_WORLD_EFI""$HALLO_WORD_EFI""$MY_ACPIVIEW_EFI")
RESOURCE_LIST=("$HELLO_WORLD_EFI")
for RESOURCE in "${RESOURCE_LIST[@]}"; do
    echo "检查文件: $RESOURCE"
    if [ ! -f "$RESOURCE" ]; then
        echo "错误：$RESOURCE 不存在，请确认编译路径是否正确"
        exit 1
    fi
done

#创建运行目录（如果不存在）
PLAYGROUND_DIR="/home/czx/kvm/project1/hello"

#创建目录
#rm -rf "$PLAYGROUND_DIR"
#mkdir -p "$PLAYGROUND_DIR"
#复制OVMF变量文件（避免修改原始文件）
cp "$OVMF_VARS" "${PLAYGROUND_DIR}/OVMF_VARS.fd"
#mkdir -p "$PLAYGROUND_DIR/uefi"
#复制资源文件到运行目录
cp "$HELLO_WORLD_EFI" "${PLAYGROUND_DIR}/uefi/HelloWorld.efi"
cp "$MY_ACPIVIEW_EFI" "${PLAYGROUND_DIR}/uefi/MyACPI.efi"
cp "$Change_ACPI_EFI" "${PLAYGROUND_DIR}/uefi/ChangeACPI.efi"
cp "$New_ACPI_EFI" "${PLAYGROUND_DIR}/uefi/NewACPI.efi"
cp "$BZIMAGE" "${PLAYGROUND_DIR}/uefi/bzImage.efi"
cp "$INITRD" "${PLAYGROUND_DIR}/uefi/initramfs.cpio.gz"
#暂停
#read-p"按任意键继续..."
#启动QEMU进入UEFI shell
qemu-system-x86_64 -machine q35,accel=kvm -m 8G -smp 4 -drive if=pflash,format=raw,unit=0,file="${OVMF_CODE}",readonly=on -drive if=pflash,format=raw,unit=1,file="${PLAYGROUND_DIR}/OVMF_VARS.fd" -drive file=fat:rw:"${PLAYGROUND_DIR}/uefi",format=raw,if=ide,index=0 -nographic -no-reboot -serial mon:stdio