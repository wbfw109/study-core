#!/bin/bash
### fish shell settings
sudo apt-add-repository -y ppa:fish-shell/release-3
sudo apt update -y
sudo apt install -y fish

# Define the Fish configuration file path
FISH_CONFIG="$HOME/.config/fish/config.fish"

# Create the directory if it doesn't exist
mkdir -p "$(dirname "$FISH_CONFIG")"




### update and upgrade packages
sudo apt update -y && sudo apt upgrade -y


##### Custom commands
sudo apt install -y curl


########################################
#### Run init_in_fish.fish
fish init_in_fish.fish
