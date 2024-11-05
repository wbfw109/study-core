#!/usr/bin/env fish
# Written at 📅 2024-10-30 15:39:12
: '
# https://gee6809.github.io/posts/qemu-network/
✈️ Purpose: Configure Bridge / Tap Network. not NAT (Network Address Translation).
                      +---------------------+ 
                      |   Host Machine      | 
                      |                     | 
                      |                     | 
                      |     +-----------+   | 
      LAN  --------------------- eth0   |   |       +-------------------+
(192.168.0.1)         |     |   tap0 ---------------|        VM0        |
                      |     |   tap1 ------------+  |  (192.168.0.3)    |
                      |     +-----------+   |    |  +-------------------+
                      |         br0         |    |
                      |    (192.168.0.2)    |    |  +-------------------+
                      |                     |    +--|        VM1        |
                      +---------------------+       |  (192.168.0.4)    |
                                                    +-------------------+
Tap serves to direct network data flow in another direction.
Bridge acts like a virtual switch and connects networks together.
📝 Taps and bridges must be created on the host machine.


About 🪱 "connection.autoconnect yes"
    Enable autoconnect to ensure that this network connection is automatically activated whenever possible.
    This is crucial for persistent network setups, especially for bridge interfaces, as it ensures that the connection remains active even after system reboots.
    Without autoconnect, manual intervention would be required to bring the network up after each reboot.
'
# Include Modules
set FISH_MODULES_PATH prototypes/_initialization/ubuntu/fish_modules
source $FISH_MODULES_PATH/argparse_utils.fish




# Handle SIGINT (Ctrl+C) to exit the script and terminate any child processes
function on_interrupt
    echo -e "\nScript interrupted. Exiting..."
    # Kill all child processes in the same process group
    kill -- -$fish_pid
    exit 1
end
trap on_interrupt SIGINT


# Constants
set SERVICE_PREFIX run_qemu-
set QEMU_DIR "$HOME/qemu"

# Usage function: Display help message
function usage
    echo "Usage: run_qemu.fish <main_command> <subcommands>..."
    echo "Main Commands:"
    echo "  vm init <vm_id> (-o --os) <os> (-v | --version) <version> (-t | --type) <type> (-a | --arch) <arch>"
    echo "    - vm_id: Virtual Machine ID (unique identifier for each VM instance)"
    echo "    - os: The operating system, e.g., 'ubuntu'"
    echo "    - version: OS version, e.g., '22'"
    echo "    - type: System type, e.g., 'desktop' or 'server'"
    echo "    - arch: Architecture, e.g., 'arm64', 'amd64' (default is 64-bit)"
    echo "    Initializes a VM, downloads the necessary image, and creates a VHD file named '<vm_id>.qcow2'."
    echo ""
    echo "  vm id <vm_id> (-r | --role) <role> <action>"
    echo ""
    echo "  vm id <vm_id> -r server set cloud-init <action>"
    echo "    - action: One of 'init', 'update'"
    echo "  vm id <vm_id> -r server run (-n | --network-id) <tap_name>"
    echo "  vm id <vm_id> -r server stop"
    echo ""
    echo "  vm id <vm_id> -r client <action>"
    echo "    - action: One of 'run', 'stop'"
    echo ""
    echo "  network init <network_id>"
    echo "    Initializes the network configuration for the specified network_id."
    echo "  network id <network_id> recover"
    echo "    Restores network configuration to its original state for the specified network_id."
    echo "  network check"
    echo "    Checks the status of all network services created by this script."
    echo "  network list"
    echo "    Lists all network services created by this script."
    echo ""
    echo "Examples:"
    echo "  run_qemu.fish vm init yocto_vm -o ubuntu -v 22 -t desktop -a amd64"
    echo "  run_qemu.fish vm id yocto_vm -r server run -n yocto_net-tap0"
    echo "  run_qemu.fish network init yocto_net"
    echo "  run_qemu.fish network id yocto_net recover"
end

# Main command validation
if test (count $argv) -lt 2
    usage
    exit 1
end

# 🔡 Snippet 📅 2024-11-04 21:15:06
check_argv_or_usage_and_exit $argv[1]
set main_command $argv[1]
set --erase argv[1]

check_argv_or_usage_and_exit $argv[1]
set sub_command $argv[1]
set --erase argv[1]

switch $main_command
    case vm
        switch $sub_command
            case init
                check_argv_or_usage_and_exit $argv[1]
                set vm_id $argv[1]
                set --erase argv[1]

                # 🔡 Snippet 📅 2024-11-04 21:01:37
                argparse 'o/os=!test -n "$_flag_value"' \
                    'v/version=!test -n "$_flag_value"' \
                    't/type=!test -n "$_flag_value"' \
                    'a/arch=!test -n "$_flag_value"' -- $argv
                or begin
                    echo "❓ Error: Invalid arguments for init command."
                    usage; and exit 1
                end
                # Check if required options are provided
                if not set -q _flag_os; or not set -q _flag_version; or not set -q _flag_type; or not set -q _flag_arch
                    echo "❓ Error: All options (-o, -v, -t, -a) are required for vm init."
                    usage; and exit 1
                end

                echo "Initializing VM with ID: $vm_id, OS: $_flag_os, Version: $_flag_version, Type: $_flag_type, Arch: $_flag_arch"
                # Additional initialization logic here
                exit 0

            case id
                check_argv_or_usage_and_exit $argv[1]
                set vm_id $argv[1]
                set --erase argv[1]

                argparse --ignore-unknown 'r/role=!test -n "$_flag_value"' -- $argv
                or begin
                    echo "❓ Error: Invalid arguments for id command."
                    usage; and exit 1
                end
                # Check if required options are provided
                if not set -q _flag_role
                    echo "❓ Error: Role ('server' | 'client') is required for vm init."
                    usage; and exit 1
                end

                check_argv_or_usage_and_exit $argv[1]
                set action $argv[1]
                set --erase argv[1]

                switch $_flag_role
                    case server
                        switch $action
                            case set
                                check_argv_or_usage_and_exit $argv[1]
                                set property $argv[1]
                                set --erase argv[1]

                                check_argv_or_usage_and_exit $argv[1]
                                set cloud_init_action $argv[1]
                                set --erase argv[1]

                                if test $property = cloud-init
                                    switch $cloud_init_action
                                        case init
                                            echo "Initializing cloud-init..."
                                        case update
                                            echo "Updating cloud-init..."
                                        case '*'
                                            usage
                                            exit
                                    end
                                    # Additional logic for cloud-init
                                else
                                    usage
                                    exit 1
                                end

                            case run
                                argparse 'n/network-id=!test -n "$_flag_value"' -- $argv
                                or begin
                                    echo "❓ Error: Invalid arguments for server run command."
                                    usage; and exit 1
                                end
                                # Check if required options are provided
                                if not set -q _flag_network_id
                                    echo "❓ Error: Network ID (-n or --network-id) is required for 'server run' command."
                                    usage; and exit 1
                                end

                                echo "Running server $vm_id with network: $_flag_network_id"
                                # Run server logic here

                            case stop
                                echo "Stopping server $vm_id"
                                # Stop server logic here

                            case '*'
                                usage
                        end

                    case client
                        switch $action
                            case run
                                echo "Running client $vm_id"
                                # Run client logic
                            case stop
                                echo "Stopping client $vm_id"
                                # Stop client logic
                            case '*'
                                usage
                        end

                    case '*'
                        usage
                end
        end

    case network
        switch $sub_command
            case init
                check_argv_or_usage_and_exit $argv[1]
                set network_id $argv[1]
                set --erase argv[1]

                echo "Initializing network with ID: $network_id"
                # Network initialization logic here
                exit 0

            case id
                check_argv_or_usage_and_exit $argv[1]
                set network_id $argv[1]
                set --erase argv[1]

                check_argv_or_usage_and_exit $argv[1]
                set action $argv[1]
                set --erase argv[1]

                switch $action
                    case recover
                        echo "Recovering network configuration for $network_id"
                        # Network recovery logic here
                        exit 0

                    case '*'
                        echo "❓ Error: Invalid subcommand for 'network id <network_id>'"
                        usage
                        exit 1
                end

            case check
                echo "Checking network status..."
                # Network status checking logic here
                exit 0

            case list
                echo "Listing all network services created by this script:"
                systemctl list-units --type=service --no-pager | awk '{print $1}' | grep "$SERVICE_PREFIX" | sed "s/^$SERVICE_PREFIX//;s/-setup.service\$//"
                exit 0

            case '*'
                echo "❓ Error: Invalid action for network command"
                usage
                exit 1
        end

    case '*'
        usage
end


: '
❔ About qemu run Options 📰 TODO:

'