##### configure_bridge_network_to_vm.fish
#!/usr/bin/env fish
# Written at 📅 2024-10-30 15:39:12


set connection_name 'eth-br0'
set bridge_name 'br0'
set tap_name 'tap0'

# ──────────────────────────────────────────────────────────────────────
# Modules and utility imports
set FISH_MODULES_PATH prototypes/_initialization/ubuntu/fish_modules
source $FISH_MODULES_PATH/network_utils.fish
set DEFAULT_ETHERNET_INTERFACE_NAME (list_physical_interfaces)

echo "Your default ethernet physic interface is '$DEFAULT_ETHERNET_INTERFACE_NAME'!"

# Safety: Handle interruptions gracefully
trap on_interrupt SIGINT

# ──────────────────────────────────────────────────────────────────────
# Usage functions
function usage
    echo "Usage: configure_bridge_network_to_vm.fish <command>"
    echo "Commands:"
    echo "  network <subcommand>"
    echo "    init     - Initialize network bridge and TAP interface"
    echo "    recover - Recover network to original state"
    echo "    check    - Display network status"
    echo ""
    echo "  vm <subcommand>"
    echo "    init         - Set up VM environment and download images"
    echo "    ubuntu init  - Initialize and run an Ubuntu VM with a virtual disk"
    exit 1
end

function network_usage
    echo "Usage: configure_bridge_network_to_vm.fish network <subcommand>"
    echo "Subcommands:"
    echo "  init     - Set up a bridge network and TAP interface"
    echo "  recover - Revert network changes"
    echo "  check    - Check current network status"
    exit 1
end

function vm_usage
    echo "Usage: configure_bridge_network_to_vm.fish vm <subcommand>"
    echo "Subcommands:"
    echo "  init        - Prepare VM environment and download required files"
    echo "  ubuntu init - Create and run an Ubuntu VM with a virtual disk"
    exit 1
end
# ──────────────────────────────────────────────────────────────────────
# Unit functionss for low level interface
function setup_tap_interface
    echo "Setting up TAP interface..."

    # By running this command, the network connection previously handled by the $DEFAULT_ETHERNET_INTERFACE_NAME (e.g., eno1) will be replaced by the bridge interface ($connection_name).
    # You can verify the change using the `ip -c a` command.
    # Be cautious if you are working remotely, as the connection may drop momentarily during this transition.
    # 📝 Note: No need to manually bring up '$bridge_name', as activating '$connection_name' automatically handles the bridge interface activation.
    
    sudo nmcli connection up $bridge_name
    ## TAP setup (one-time. not maintained aftger reboot)
    # https://ahelpme.com/linux/centos-8/create-bridge-and-add-tun-tap-device-using-networkmanager-nmcli-under-centos-8/
    # 1. Create a TAP interface
    sudo ip tuntap add dev $tap_name mode tap user (whoami)

    # 2. Attach the TAP interface to the bridge
    sudo ip link set $tap_name master $bridge_name

    # 3. Activate the TAP interface
    # It may take up to 10 seconds for the TAP interface to initialize inside a VM.
    sudo ip link set $tap_name up

    echo "TAP interface $tap_name set up and attached to bridge $bridge_name."
end

# 2. Function to clean up the TAP interface
function cleanup_tap_interface
    echo "Cleaning up TAP interface..."

    # 1. Deactivate and delete the TAP interface
    sudo ip link set $tap_name down
    sudo ip tuntap del dev $tap_name mode tap

    echo "TAP interface $tap_name deleted."
end

# ──────────────────────────────────────────────────────────────────────
# Service-related functions

# 1. Create a systemd service for TAP interface setup
function create_tap_service
    echo "Creating TAP setup service..."

    set SERVICE_NAME "tap-setup.service"
    set SERVICE_PATH "/etc/systemd/system/$SERVICE_NAME"

    # Write the service using tee
    sudo bash -c "tee $SERVICE_PATH > /dev/null <<EOF
[Unit]
Description=TAP Interface Setup Service
# Ensure the service starts only after basic network is ready and NetworkManager is active.
After=network.target NetworkManager.service
# Prefer to run this service when the network is fully online, but it's not mandatory.
Wants=network-online.target

[Service]
Type=oneshot
ExecStart=/usr/bin/env bash -c ' \
  sudo nmcli connection up $bridge_name && \
  sudo ip tuntap add dev $tap_name mode tap user $(whoami) && \
  sudo ip link set $tap_name master $bridge_name && \
  sudo ip link set $tap_name up \
'
ExecStop=/usr/bin/env bash -c ' \
  sudo ip link set $tap_name down && \
  sudo ip tuntap del dev $tap_name mode tap \
'
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF"

    echo "Reloading systemd and enabling the TAP service..."
    sudo systemctl daemon-reload
    sudo systemctl enable $SERVICE_NAME
    sudo systemctl start $SERVICE_NAME
    echo "$SERVICE_NAME started and enabled successfully."
end



# 2. Stop and disable the TAP service
function remove_tap_service
    echo "Stopping and disabling TAP setup service..."

    set SERVICE_NAME "tap-setup.service"

    sudo systemctl stop $SERVICE_NAME
    sudo systemctl disable $SERVICE_NAME
    sudo rm /etc/systemd/system/$SERVICE_NAME
    sudo systemctl daemon-reload

    echo "$SERVICE_NAME stopped, disabled, and removed successfully."
end

# ──────────────────────────────────────────────────────────────────────
# Network-related functions
function network_init
    echo "Initializing bridge network..."

    # 1. Create a bridge interface
    sudo nmcli connection add type bridge ifname $bridge_name con-name $bridge_name

    # 2. Set static IP address and other configurations on the bridge
    sudo nmcli connection modify $bridge_name ipv4.addresses 10.10.14.19/24
    sudo nmcli connection modify $bridge_name ipv4.gateway 10.10.14.254
    sudo nmcli connection modify $bridge_name ipv4.dns 203.248.252.2
    sudo nmcli connection modify $bridge_name ipv4.method manual
    sudo nmcli connection modify $bridge_name ipv4.ignore-auto-routes yes
    sudo nmcli connection modify $bridge_name ipv4.ignore-auto-dns yes
    sudo nmcli connection modify $bridge_name connection.autoconnect yes
    sudo systemctl restart NetworkManager

    # 3. Add Ethernet interface to the bridge
    sudo nmcli connection add type ethernet slave-type bridge \
        con-name $connection_name ifname $DEFAULT_ETHERNET_INTERFACE_NAME master $bridge_name
    sudo nmcli connection modify eno1-br0 connection.autoconnect yes

    # 4. Activate the bridge connection
    sudo nmcli connection up $connection_name

    # 5. Setup the TAP interface
    create_tap_service

    echo "Bridge network initialized. Verify with 'ip -c a'."
end


function network_recover
    : '
    📰📅 2024-10-29 16:44:23
        I am unable to solve the issue of deactivating and deleting a connection continuously without using explicit sleep commands.  
        It appears to be caused by the asynchronous behavior of the `nmcli connection` command, which may not wait for the state change to complete before proceeding.
        To use "--wait" argument not works well.
    '

    echo "Recovering network configuration..."

    # 1. Clean up TAP interface
    remove_tap_service

    # 2. Deactivate the bridge connection
    sudo nmcli connection down $connection_name
    sleep 2

    # 3. Delete the Ethernet connection from the bridge
    sudo nmcli connection delete $connection_name

    # 4. Deactivate and delete the bridge interface
    sudo nmcli connection down $bridge_name
    sleep 2
    sudo nmcli connection delete $bridge_name

    # 5. Restart NetworkManager to ensure a clean state
    sudo systemctl restart NetworkManager

    echo "Network recovery completed."
end


function network_check
    echo "Checking network status..."
    echo -e '\nAll IP addresses of network interfaces:'
    ip -c a

    echo -e '\nAll network links:'
    ip -c l

    echo -e '\nAll network routes:'
    ip -c r
end

# ──────────────────────────────────────────────────────────────────────
# VM-related functions
function vm_init
    echo "⚠️  Warning: This operation will download images and may overwrite existing files in '~/qemu/'."
    echo "Do you want to proceed? (y/n)"
    read -l confirmation

    switch $confirmation
        case 'y'
            echo "Proceeding with VM environment initialization..."
            echo "Downloading Ubutnu 22.04 Desktop and Server .iso..."

            ## Images
            # Supported platforms: https://ubuntu.com/core/docs/supported-platforms
            # Ubuntu 22 images: https://releases.ubuntu.com/jammy/
            # Images are compressed with xz; decompress with: xz -d <image-name>.img.xz

            # Download the desktop and server ISO images
            wget -P ~/qemu https://releases.ubuntu.com/jammy/ubuntu-22.04.5-desktop-amd64.iso
            wget -P ~/qemu https://releases.ubuntu.com/jammy/ubuntu-22.04.5-live-server-amd64.iso

            # Install QEMU and required dependencies
            sudo apt update && sudo apt install -y \
                qemu-system qemu-kvm libvirt-daemon-system bridge-utils virtinst ovmf

            # https://www.qemu.org/download/
            # Open Virtual Machine Firmware (OVMF) ; https://wiki.ubuntu.com/UEFI/OVMF
            echo "Copy OVMF firmware files for UEFI boot..."
            cp /usr/share/OVMF/OVMF_CODE_4M.secboot.fd ~/qemu
            cp /usr/share/OVMF/OVMF_VARS_4M.ms.fd ~/qemu

            echo "VM environment initialized."

        case 'n'
            echo "Operation canceled. Exiting without changes."
            return

        case '*'
            echo "Invalid input. Please enter 'y' or 'n'."
            vm_init  # Re-prompt the user if the input is invalid
    end
end


function vm_ubuntu_init
    echo "Creating virtual disk and running Ubuntu VM..."

    ## ISO files are generally considered read-only media and cannot be modified.
    # They emulate CD/DVD images, so it is not possible to change or save data on them.
    # Therefore, if the VM is run with only an ISO file, any changes made will not persist after the VM is restarted.
    # To retain changes, a writable virtual disk file is required.
    # 🕥 Testing with Quick Emulator with virtual disk file ; https://ubuntu.com/server/docs/virtualisation-with-qemu
    # https://documentation.ubuntu.com/server/how-to/virtualisation/libvirt/
    # Ubuntu Core with QEMU Without TPM emulation

    echo "⚠️  Warning: This process will create a new virtual disk to '$HOME/qemu/ubuntu-vm.qcow2 150G' and launch the VM."
    echo "Do you want to proceed? (y/n)"
    read -l confirmation

    switch $confirmation
        case 'y'
            echo "Proceeding with VM creation and initialization..."

            echo "Create the virtual disk file..."
            qemu-img create -f qcow2 $HOME/qemu/ubuntu-vm.qcow2 150G

            echo "Launch the VM with the specified configuration..."
            qemu-system-x86_64 \
                -enable-kvm -smp 1 -m 2048 -machine q35 -cpu host \
                -global ICH9-LPC.disable_s3=1 \
                -net nic,model=virtio \
                # -net user # NAT with PortForwarding \
                -net user,hostfwd=tcp::8022-:22,hostfwd=tcp::8090-:80 \
                # -net user # NAT \
                -drive file=$HOME/qemu/OVMF_CODE_4M.secboot.fd,if=pflash,format=raw,unit=0,readonly=on \
                -drive file=$HOME/qemu/OVMF_VARS_4M.ms.fd,if=pflash,format=raw,unit=1 \
                -drive file=$HOME/qemu/ubuntu-vm.qcow2,if=none,id=disk0,format=qcow2,cache=writeback \
                -device virtio-blk-pci,drive=disk0,bootindex=0 \
                -drive file=$HOME/qemu/ubuntu-22.04.5-desktop-amd64.iso,if=none,id=cdrom,media=cdrom \
                -device ide-cd,bus=ide.1,drive=cdrom \
                -serial mon:stdio

            echo "Ubuntu VM initialized."

        case 'n'
            echo "Operation canceled. Exiting without changes."
            return

        case '*'
            echo "Invalid input. Please enter 'y' or 'n'."
            vm_ubuntu_init  # Re-prompt the user if input is invalid
    end
end

# ──────────────────────────────────────────────────────────────────────
# Main command dispatch
if test (count $argv) -eq 0
    usage
end

switch $argv[1]
    case 'network'
        if test (count $argv) -lt 2
            network_usage
        end

        switch $argv[2]
            case 'init'
                network_init
            case 'recover'
                network_recover
            case 'check'
                network_check
            case '*'
                network_usage
        end

    case 'vm'
        if test (count $argv) -lt 2
            vm_usage
        end

        switch $argv[2]
            case 'init'
                vm_init
            case 'ubuntu'
                if test (count $argv) -ge 3 -a "$argv[3]" = 'init'
                    vm_ubuntu_init
                else
                    vm_usage
                end
            case '*'
                vm_usage
        end

    case '*'
        usage
end

