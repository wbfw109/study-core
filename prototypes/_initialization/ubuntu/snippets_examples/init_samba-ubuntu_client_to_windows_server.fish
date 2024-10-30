# Written at 📅 2024-10-28 13:48:11
: '
* 🧪 if you already mount, you can test this script after run %shell> sudo umount $LOCAL_MOUNT_POINT
* Prerequisite
%shell> sudo apt update -y && sudo apt install -y samba
'
### 0. Install required packages, Load my modules, Define functions
# It requires sudo permissions.
sudo -v

# https://ubuntu.com/tutorials/install-and-configure-samba#2-installing-samba

# Handle SIGINT (Ctrl+C) to exit the script and terminate any child processes
function on_interrupt
    echo -e "\nScript interrupted. Exiting..."
    # Kill all child processes in the same process group
    kill -- -$fish_pid
    exit 1
end
trap on_interrupt SIGINT



## 1.1. Set the path to the remote share and the local mount point, credentai file
# Define the remote SMB/CIFS share path
set -x REMOTE_SHARE_PATH //10.10.14.40/samba

# Define the local directory where the share will be mounted
set -x LOCAL_MOUNT_POINT $HOME/mnt/class

# The credentials file name is hardcoded for class5 on Windows
set -x CREDENTIALS_FILE /etc/samba/class5_windows_credentials
mkdir --parents $LOCAL_MOUNT_POINT
# --parents: Create parent directories as needed without error if they already exist


## 1.2. Prompt the user for input (password hidden)
# Prompt for the remote username
echo "Enter remote username: "
read REMOTE_USERNAME

# Prompt for the remote password (hidden input)
echo "Enter password: "
read --silent REMOTE_PASSWORD

# Add a newline after password entry for better formatting
echo ""




## 1.3. Ensure the credentials file exists

## Write the username and password to the credentials file securely
# No need for '-e' since Fish's echo interprets escape sequences by default
echo -e "username=$REMOTE_USERNAME\npassword=$REMOTE_PASSWORD" \
  | sudo tee $CREDENTIALS_FILE > /dev/null

## Set ownership of the credentials file to the current user
# Ensures the user can access the file without sudo
sudo chown $USER:$USER $CREDENTIALS_FILE

## Verify that the credentials file was created and is accessible
if not test -f $CREDENTIALS_FILE
    echo "Error: Failed to create credentials file $CREDENTIALS_FILE."
    exit 1
end

echo "Credentials file $CREDENTIALS_FILE created successfully."




## 2.1. Extract username and password from the credentials file using grep with Perl-compatible regex
set USERNAME (grep -oP '(?<=username=).*' $CREDENTIALS_FILE)
set PASSWORD (grep -oP '(?<=password=).*' $CREDENTIALS_FILE)
# -o, --only-matching: Output only the part of the line that matches the regex.
# -P: Enable Perl-compatible regular expressions (PCRE) for advanced pattern matching.
# (?<=username=): Positive lookbehind to match only the text that follows 'username='.
# .*: Match any number of characters after 'username='.

## 2.2. Test access using Samba client by listing the contents of the shared folder
smbclient $REMOTE_SHARE_PATH -U $USERNAME%$PASSWORD -c "ls"
# smbclient: Connects to the remote share and runs commands on it
# -U: Provides the username and password in username%password format.
# -c, --command: Executes a specified command (e.g., "ls") on the remote share.




## 2.3. Mount the shared folder to the local directory using CIFS protocol
sudo mount -t cifs -o credentials=$CREDENTIALS_FILE,vers=3.0 \
  $REMOTE_SHARE_PATH $LOCAL_MOUNT_POINT

# Check the result of the mount operation
if test $status -eq 0
    echo "Mount successful. Unmounting to proceed with fstab configuration..."
    # lazy unmount
    sudo umount -l $LOCAL_MOUNT_POINT
else
    echo "Error: Mounting failed."
    exit 1
end



## 3.1. Add the mount configuration to /etc/fstab for automatic mounting at boot
#🚣 '_netdev' ensures the network is up before mounting the share.
## 3.1. Add the mount configuration to /etc/fstab for automatic mounting at boot



set fstab_line "$REMOTE_SHARE_PATH $LOCAL_MOUNT_POINT cifs credentials=$CREDENTIALS_FILE,vers=3.0,_netdev,user,uid=1000,gid=1000,file_mode=0777,dir_mode=0777 0 0"
if not grep --quiet --fixed-strings "$fstab_line" /etc/fstab
    echo "$fstab_line" | sudo tee -a /etc/fstab > /dev/null
    systemctl daemon-reload  # Reloads systemd to apply changes to fstab.
    echo "Mount configuration added to /etc/fstab."
else
    echo "Configuration already exists in /etc/fstab. Skipping."
end



## 3.2. Reload systemd to apply the fstab changes
echo "Reloading systemd to apply fstab changes..."



## 3.3. Check if the shared folder is already mounted
echo "Checking if the shared folder is already mounted..."

if mount | grep -q "$LOCAL_MOUNT_POINT"
    echo "Shared folder is already mounted at $LOCAL_MOUNT_POINT."
else
    echo "Shared folder is not mounted. Attempting to mount now..."
    sudo mount $LOCAL_MOUNT_POINT

    if test $status -eq 0
        echo "Shared folder mounted successfully from fstab configuration."
    else
        echo "Error: Failed to mount the shared folder."
        exit 1
    end
end
: '
💡 Explanation of the fstab configuration line being added:
# https://help.ubuntu.com/community/Fstab
<REMOTE_SHARE_PATH> <LOCAL_MOUNT_POINT> <file_system_type> <options> <dump> <pass>

1. REMOTE_SHARE_PATH:
   - This refers to the remote CIFS (Windows) share you want to mount.
     Example: //10.10.14.40/samba

2. LOCAL_MOUNT_POINT:
   - This is the local directory where the remote share will be mounted.
     Example: /home/wbfw109v2/mnt/class

3. File System Type: 
   - "cifs" indicates that the network file system is using the CIFS (SMB) protocol.

4. Options: 
   - This part includes several comma-separated options to control the mount behavior.

  - credentials=<path>: 
    - Points to a file containing the username and password required to access the remote share.
    - Example: /etc/samba/class5_windows_credentials

  - vers=3.0:
    - Specifies the SMB protocol version to use for compatibility. 
      SMB 3.0 is generally used for modern Windows systems.

  -🚣 _netdev:
    - Ensures that the network device is available before attempting to mount the share.
      This prevents boot-time mount failures caused by missing network connections.

  -🚣 user:
    - Allows the mount to be performed by non-root users. 
      This is essential for GUI-based file managers or regular users to access the mount.

  - uid=1000, gid=1000:
    - Sets the user ID (UID) and group ID (GID) for all files and directories inside the mount.
      These should match the UID and GID of the user accessing the mount to avoid permission issues.

  - file_mode=0777, dir_mode=0777:
    - Sets the permissions for all files and directories within the mounted share.
      "0777" grants read, write, and execute permissions to all users (owner, group, others).
      Adjust these permissions based on your security requirements.

  -🚣 x-systemd.automount:
    - Enables on-demand mounting. 
      The share will only be mounted when it is accessed, reducing boot time and preventing unnecessary mounts.

5. Dump (0):
   - Controls whether the filesystem should be dumped (backed up) by the dump utility. 
     A value of "0" disables this feature.

6. Pass (0):
   - Controls the order in which filesystems are checked at boot by the fsck utility.
     A value of "0" means the filesystem will not be checked.

'


# ### 4. Add the bookmarks in the 
# 📝 It is not requried because mount directory already attach to File Manager Bookmarks.
# # Load my modules
# set -x FISH_MODULES_PATH prototypes/_initialization/ubuntu/fish_modules
# echo "Adding LOCAL_MOUNT_PATH to Ubuntu File Manager Bookmarks"
# source $FISH_MODULES_PATH/bookmark_utils.fish

# # Attempt to add each path to the bookmarks file
# add_bookmark $LOCAL_MOUNT_POINT

# # Check the result of systemd reload
# if test $status -eq 0
#     echo "Bookmark update completed!"
# else
#     echo "Error: Failed to update bookmark."
#     exit 1
# end
