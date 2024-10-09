#!/usr/bin/env fish
# Written at 📅 2024-10-04 16:37:59

# Function to fix GPG keys for a given repository and key URL
function fix_gpg_key
  set repo_url $argv[1]
  set key_url $argv[2]
  set repo_file (string replace -r "https?://" "" "$repo_url" | string replace "/" "-" | string replace ":" "-")
  echo "Fixing GPG key for $repo_url..."

  # Create keyring directory if it doesn't exist
  sudo mkdir -p /etc/apt/keyrings

  # Download and store the GPG key
  curl --fail --silent --show-error --location "$key_url" | gpg --dearmor | sudo tee /etc/apt/keyrings/$repo_file.gpg > /dev/null

  # Update the repository entry with the new key
  echo "deb [signed-by=/etc/apt/keyrings/$repo_file.gpg] $repo_url $(lsb_release -cs) main" | \
      sudo tee /etc/apt/sources.list.d/$repo_file.list > /dev/null

  echo "GPG key fixed for $repo_url."
end

# Run apt update and capture output
set apt_output (sudo apt update 2>&1)

# Check for deprecation warnings or legacy GPG keys in the trusted.gpg keyring
for line in (echo $apt_output | string split '\n')
  if echo $line | grep -q 'legacy trusted.gpg keyring'
      # Extract the repository URL from the error line
      set repo_url (echo $line | grep -o 'https?://[^ ]+' | head -n 1)

      # Add known GPG key URLs here based on repo_url
      switch "$repo_url"
          case '*packages.microsoft.com*'
              set key_url 'https://packages.microsoft.com/keys/microsoft.asc'
          case '*apt.repos.intel.com*'
              set key_url "https://apt.repos.intel.com/openvino/gpgkey"
          case '*dl.google.com*'
              set key_url 'https://dl.google.com/linux/chrome-remote-desktop/deb/gpg'
          case '*deb.anydesk.com*'
              set key_url 'https://keys.anydesk.com/repos/DEB-GPG-KEY'
          case '*dl.winehq.org*'
              set key_url 'https://dl.winehq.org/wine-builds/winehq.key'
          case '*'
              echo "Unknown repo $repo_url. You may need to manually find the key URL."
              continue
      end

      # Fix the GPG key for the repository
      fix_gpg_key "$repo_url" "$key_url"
  end
end

# Update the package lists again after fixing keys
echo "Updating package lists again..."
sudo apt-get update


## You may need to run additional commands if you encounter the following error messages:
# 🛍️ e.g. W: An error occurred during the signature verification. The repository is not updated, and the previous index files will be used. 
# GPG error: https://apt.repos.intel.com/openvino/2023 ubuntu22 InRelease: The following signatures couldn't be verified because the public key is not available: NO_PUBKEY BAC6F0C353D04109
#
# In that case, follow these steps:
# 1. Run the following command to search for the repository file:
#    %shell> grep -r "openvino" /etc/apt/
#
# 2. Once you find the file, remove it using:
#    %shell> sudo rm /etc/apt/sources.list.d/intel-openvino.list
#
# This will resolve the GPG key issue related to Intel OpenVINO.
