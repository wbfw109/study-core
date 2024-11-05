##### temp.fish
#!/usr/bin/env fish
# 📅 2024-10-30 20:00:16
set usb_info (lsusb | grep -i CAMERA)
if test -z $usb_info
    echo "Not found valid Camera device"
    exit 1
end

set bus_id (echo $usb_info | awk '{print $2}')
set device_id (echo $usb_info | awk '{print $4}' | tr --delete ':')

set vendor_id (echo $usb_info | awk '{print $6}' | cut --delimiter ':' --fields=1)
set product_id (echo $usb_info | awk '{print $6}' | cut --delimiter ':' --fields=2)
# echo "Bus ID: $bus_id"
# echo "Device ID: $device_id"
# echo "Vendor ID: $vendor_id"
# echo "Product ID: $product_id"

: '
## TODO:

Note that Default path of All files is variable: workspace_distribution: ~/qemu/<distribution>

mkdir -p <all supported distribution>. current list: ubuntu_22_desktop_amd64. the distribution name will be complete by user comman like "ubuntu --version 22 --type desktp --arch amd64 ..."

Comandline fish script: run_qemu_vm


Available command. Each arguments will 
    vm ubuntu --version 22 --type desktp --arch amd64 server init vhd
        >> it will download target .iso file to workspace_distribution.(if the image already exists, read "y" or "n" and redownload or skip.). each urls is referenced by varaiable. e.g. set ubuntu_22_desktop_amd64_iso_url https://releases.ubuntu.com/jammy/ubuntu-22.04.5-desktop-amd64.iso
    vm ubuntu --version 22 --type desktp --arch amd64 server init cloud-init
        >> it will run create user-data.yaml meta-data.yaml, network-config.yaml files and run %shell> cloud-localds cloud-init.iso user-data.yaml meta-data.yaml --network-config=network-config.yaml
    vm ubuntu --version 22 --type desktp --arch amd64 server update cloud-init
        >> cloud-localds cloud-init.iso user-data.yaml meta-data.yaml --network-config=network-config.yaml
     22 --type desktp --arch amd64 server run
        >> variable target_vhd = vhd.qcow2                   // string concat with "_" delimiter. and all these arguments must be not none.
        >> variable spice_socket = spice.sock                // string concat with "_" delimiter. and all these arguments must be not none.
        >> it will run %shell> qemu-system-x86_64 ...
    vm ubuntu --version 22 --type desktp --arch amd64 server stop
        it will kill the qemu of the distrubition. pid can be obtain by ps aux | grep qemu | grep <distribution> | awk ... print $1"
    vm ubuntu --version 22 --type desktp --arch amd64 client run
        >> it will run %shell> remote-viewer spice+unix://$spice_socket
    vm ubuntu --version 22 --type desktp --arch amd64 client stop
        it will kill the remote viewer of the distrubition. pid can be obtain by ps aux | grep remote-viewer | grep <distribution>" | awk ... print $1

        
    network init
        >> it will .. only change my host ethernet to bridge network. refer to "prototypes/_initialization/ubuntu/howto/configure_bridge_network_to_vm.fish"


variables
    target_vhd: The target virtual disk varies depending on the argument specified by the user. format: <os>_<version>_<type>_<arch>
    run_binary: must be changed by argument that user speicify. e.g. if amd64, run_binary will be "qemu-system-x86_64", and if arm, will be "qemu-system-aarch64".

Tip
    - not run with "sudo" to use socket "/run/user/1000/spice.sock"
    - remote-viewer automatically create and remove socket file. so you not need to manually manage the socket file.



'

qemu-system-x86_64 \
    -enable-kvm \
    -smp 1 \
    -m 2048 \
    -machine q35 \
    -cpu host \
    -netdev tap,id=net0,ifname=tap0,script=no,downscript=no \
    -device virtio-net-pci,netdev=net0,mac=52:54:00:12:34:56 \
    -global ICH9-LPC.disable_s3=1 \
    -drive file=$HOME/qemu/OVMF_CODE_4M.secboot.fd,if=pflash,format=raw,unit=0,readonly=on \
    -drive file=$HOME/qemu/OVMF_VARS_4M.ms.fd,if=pflash,format=raw,unit=1 \
    -cdrom cloud-init.iso \
    -boot d \
    -drive file=$HOME/qemu/ubuntu-vm.qcow2,if=none,id=disk0,format=qcow2,cache=writeback \
    -device virtio-blk-pci,drive=disk0,bootindex=1 \
    ## Spice ~
    -spice disable-ticketing=on \
    # GL acceleration (virgl). you can not use '-spice port=3001,disable-ticketing=on' and 'remote-viewer spice://localhost:3001'.
    -device virtio-vga-gl -spice gl=on,unix=on,addr=/run/user/1000/spice.sock \
    # Spice 🔪 Multiple monitor support. if you want to, add '-device qxl'
    -vga qxl \
    # Spice 🔪 Agent
    -device virtio-serial \
    -chardev spicevmc,id=vdagent,debug=0,name=vdagent \
    -device virtserialport,chardev=vdagent,name=com.redhat.spice.0 \
    #📰 Spice 🔪 USB redirection
    -device qemu-xhci,id=usb \
    -chardev spicevmc,name=usbredir,id=usbredirchardev1 \
    -device usb-redir,chardev=usbredirchardev1,id=usbredirdev1 \
    -chardev spicevmc,name=usbredir,id=usbredirchardev2 \
    -device usb-redir,chardev=usbredirchardev2,id=usbredirdev2 \
    -chardev spicevmc,name=usbredir,id=usbredirchardev3 \
    -device usb-redir,chardev=usbredirchardev3,id=usbredirdev3 \
    ## Run in Forground or Background: [-serial mon:stdio, -daemonize]
    -daemonize
    #📰 Spice 🔪 TLS
    #📰 Spice 🔪 Intel’s GVTg
    #📰 Spice 🔪 Video Streaming
    ## will be reload spice...
    # -device qemu-xhci,id=xhci \
    #-device usb-host,bus=xhci.0,hostbus=$bus_id,hostaddr=$device_id
# Ticketing is a simple authentication system which enables you to set simple tickets to a VM. Client has to authenticate before the connection can be established. See the Spice option password in the following examples.
#📰 wbfw109v2@iot4-computer ~/r/intel-edge-academy-6 (main)> fish temp.fish
# qemu-system-x86_64: SPICE GL support is local-only for now and incompatible with -spice port/tls-port
####### 
# TODO: Spice server supports the QXL VDI interface. QXL, SFALIC, GLZ, Lempel-Ziv (LZ) algorithm, Both Quic and LZ are local algorithms...
## client$ remote-viewer spice://myhost:3001
# remote-viewer spice+unix:///run/user/1000/spice.sock
# https://www.spice-space.org/api/spice-gtk/SpiceUsbDeviceManager.html#SpiceUsbDeviceManager--auto-connect-filter

## 🤬 manullay run: click 'music' icon (media) on the top-left of QEMU from remote viewer; 
#   'Select USB devices for redirection'
# ❓ cloud-localds; cloud local data source
# ❓ https://en.wikipedia.org/wiki/USB#Device_classes
# ❓ VGA(Video Graphics Array)
# ❓ QXL(Quick eXecuting Layer)
# ❓ UHCI (Universal Host Controller Interface): USB 1.1 규격의 컨트롤러입니다
# ❓ EHCI (Enhanced Host Controller Interface): USB 2.0 규격의 컨트롤러입니다
# ❓ ich9 (Intel I/O Controller Hub 9): Intel에서 제공하는 하드웨어 칩셋 중 하나입니다.
# sudo killall qemu-system-x86_64
# -display spice-app
### qemu-system-x86_64 -device help | grep virtio
# https://www.spice-space.org/spice-user-manual.html#spice-protocol
: '
### `-enable-kvm`
# Enables KVM (Kernel-based Virtual Machine) for hardware acceleration.
# This allows the VM to use the host CPU more efficiently, improving performance.

### `-smp 1`
# Specifies the number of CPU cores allocated to the VM. Here, only 1 core is assigned.
# 🪱 SMP: Symmetric Multiprocessing, meaning the virtual machine can use multiple CPU cores or threads in parallel if assigned.

### `-m 2048`
# Allocates 2048 MB (2 GB) of RAM to the VM for better performance.
# 🪱 The `-m` stands for "memory," specifying the amount of RAM assigned to the virtual machine.

### `-machine q35`
# Emulates a modern Intel platform with PCIe support using the Q35 chipset.

### `-cpu host`
# Passes the host CPU\'s features directly to the VM to ensure compatibility and performance.

### `-netdev tap,id=net0,ifname=tap0,script=no,downscript=no`
# Configures a TAP network interface.
# `script=no`: Disables automatic setup scripts.
# `downscript=no`: Prevents teardown scripts on shutdown.

### `-device virtio-net-pci,netdev=net0,mac=52:54:00:12:34:56`
# Adds a Virtio-based network device with a specific MAC address for performance.
# 🪱 Virtio: A virtualization interface designed for efficient I/O, reducing overhead in virtual machines.

### `-global ICH9-LPC.disable_s3=1`
# Disables S3 suspend-to-RAM state. ICH9-LPC provides legacy device support.
# 🪱 ICH9: Intel I/O Controller Hub 9; LPC: Low Pin Count. The ICH9-LPC interface supports legacy I/O devices like serial ports and PS/2 keyboards.

### `-drive file=OVMF_CODE_4M.secboot.fd,if=pflash,format=raw,unit=0,readonly=on`
# Loads UEFI firmware from a persistent flash storage.
# 🪱 OVMF: Open Virtual Machine Firmware, a project providing UEFI firmware for virtual machines. It enables UEFI boot and configuration in virtual environments.

### `-drive file=OVMF_VARS_4M.ms.fd,if=pflash,format=raw,unit=1`
# Provides writable storage for UEFI variables (e.g., NVRAM).

### `-cdrom cloud-init.iso`
# Mounts the cloud-init ISO to automate VM initialization (e.g., user setup).

### `-boot d`
# Sets the VM to boot from the CD-ROM first.

### `-drive file=ubuntu-vm.qcow2,if=none,id=disk0,format=qcow2,cache=writeback`
# Attaches the QCOW2 disk image with write-back caching for performance.
# 🪱 QCOW2: QEMU Copy-On-Write, a disk image format supporting snapshots and space-efficient storage.

### `-device virtio-blk-pci,drive=disk0,bootindex=1`
# Adds a Virtio block device with primary boot priority.
# 🪱 A Virtio block device allows fast and efficient I/O by reducing the virtualization overhead on the block device.

### `-serial mon:stdio`
# Redirects the QEMU monitor to the terminal\'s standard I/O.
# 🪱 mon:stdio: The monitor interface allows controlling the VM from the terminal for tasks like pausing or rebooting the machine.

### `-device qemu-xhci,id=xhci`
# Adds a USB 3.0 controller with backward compatibility for USB 2.0 and 1.x.
# ⚓ Extensible Host Controller Interface (xhci) ; https://en.wikipedia.org/wiki/Extensible_Host_Controller_Interface

### `-device usb-host,bus=xhci.0,hostbus=$bus_id,hostaddr=$device_id`
# Passes the host\'s USB device to the VM via the xHCI controller.

'