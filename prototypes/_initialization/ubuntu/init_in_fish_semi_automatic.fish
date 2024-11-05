#!/usr/bin/env fish
#### https://docs.usebottles.com/advanced/cli
# Define the Fish configuration file path. Ensure Path.
set -Ux FISH_CONFIG "$HOME/.config/fish/config.fish"
set -Ux FISH_COMPLETIONS "$HOME/.config/fish/completions"
set BASHRC_PATH "$HOME/.bashrc"

### Update and upgrade packages
sudo apt update -y




#### Intsall (Clipboard Manager: gnome-shell-extension-clipboard-indicator) in Ubuntu 📅 2024-10-31 19:17:00
sudo apt install gnome-shell-extensions
### 1. Available option 1 (recommneded)
# 🤬 manullay run: browser https://extensions.gnome.org/extension/779/clipboard-indicator/ and clikc the "Install"

### 1. Available option 2 (not recommneded)
# set gnone_shell_extension_dir ~/.local/share/gnome-shell/extensions
# git clone https://github.com/Tudmotu/gnome-shell-extension-clipboard-indicator.git $gnone_shell_extension_dir/clipboard-indicator@tudmotu.com
# sudo systemctl restart gdm
## 🤬 manullay run: 
# gnome-extensions enable clipboard-indicator@tudmotu.com

### 2. Change Keyboard shortcut
echo "[/]
toggle-menu=['<Super>c']
" | dconf load /org/gnome/shell/extensions/clipboard-indicator/






#### Install Wine for Kakaotalk with locale-Korean 

### install Wine ; https://gitlab.winehq.org/wine/wine/-/wikis/Debian-Ubuntu
## Preparation
sudo dpkg --add-architecture i386
cat /etc/os-release
# Download and add the repository key:
sudo mkdir -pm755 /etc/apt/keyrings
sudo wget -O /etc/apt/keyrings/winehq-archive.key https://dl.winehq.org/wine-builds/winehq.key

## Add the repository:
sudo wget -NP /etc/apt/sources.list.d/ https://dl.winehq.org/wine-builds/ubuntu/dists/noble/winehq-noble.sources
# Update the package information:
sudo apt update -y

## Install Wine
sudo apt install -y --install-recommends winehq-devel

## configure Wine to use Windows 10
winetricks -q win10


### Install Kakaotalk
wget https://app-pc.kakaocdn.net/talk/win32/KakaoTalk_Setup.exe -P ~/Downloads/
LANG="ko_KR.UTF-8" wine ~/Downloads/KakaoTalk_Setup.exe

## 🤬 manullay config: 
# ✔️ Pin the KakaoTalk program to the left Dock on the Ubuntu Desktop
# in Kakaotalk settings to prevent Korean text from breaking or displaying incorrectly
#   - settings - Display
#     - Fonts - ✔️ NanumGothic

