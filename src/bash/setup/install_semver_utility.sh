#!/usr/bin/bash
# https://github.com/fsaintjacques/semver-tool
function help()
{
  echo "'install_semver_utility.sh': install semver shell utility from https://github.com/protocolbuffers/protobuf"
  echo "❕ sudo required to move files to /usr/local/bin"
}

if [[ "$EUID" -ne 0 ]]
  then echo "❗ Please run as root"
  help
  exit
fi

wget -O /usr/local/bin/semver \
  https://raw.githubusercontent.com/fsaintjacques/semver-tool/master/src/semver

# Make script executable
chmod +x /usr/local/bin/semver

# Prove it works
semver --version