# Project Setup Instructions

📝 Note that

- This directory includes prototypes to compose new project.
- After the installation for basic settings, additional installation is required for each project.
- Its purpose is to compose project-specific isolated environment with cross-platform and cross-compiling in Windows 11.
  - But some may be different like executable extension, built-in shell commands, compiler, etc.
- the README.md written at 📅 2024-08-29 00:24:41

## Table of contents

- [Project Setup Instructions](#project-setup-instructions)
  - [Table of contents](#table-of-contents)
  - [Environment](#environment)
    - [OS](#os)
      - [In Windows](#in-windows)
        - [Packages Managed by "choco" command](#packages-managed-by-choco-command)
      - [In Ubuntu](#in-ubuntu)
        - [Packages Managed by "apt" command in "fish" shell](#packages-managed-by-apt-command-in-fish-shell)
        - [Packages Managed by "pipx" command in "fish" shell](#packages-managed-by-pipx-command-in-fish-shell)
          - [Update commands](#update-commands)
    - [1. VSCode Extension Installation](#1-vscode-extension-installation)
    - [2. Other settings](#2-other-settings)
    - [4. \[Optional\] Device settings](#4-optional-device-settings)
      - [USB camera](#usb-camera)
  - [Order of Tasks for using C++](#order-of-tasks-for-using-c)
  - [Changelog](#changelog)
  - [legacy](#legacy)
    - [Packages Managed by "pixi" command](#packages-managed-by-pixi-command)
  - [📰 Doing](#-doing)
    - [Installation in order to build with clang](#installation-in-order-to-build-with-clang)

## Environment

📝 if **\<Name\>** starts with Lower case, it is also used CLI command.

### OS

#### In Windows

| Type                        | Name       | Version                                          | Reference                                                                                                             |
| --------------------------- | ---------- | ------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------- |
| OS                          | Windows 11 | **Latest Stable;** >= 23H2 (OS Build 22631.4037) | [Download Windows 11](https://www.microsoft.com/software-download/windows11)                                          |
| Shell                       | pwsh       | **Latest Stable;** >= 7.4.5                      | [Install Powershell](https://learn.microsoft.com/en-us/powershell/scripting/install/installing-powershell-on-windows) |
| Package Manager for Windows | choco      | **Latest Stable;** >= 2.3.0                      | [Install chocolatey](https://chocolatey.org/install)                                                                  |

##### Packages Managed by "choco" command

🗝️ This tool is used to install and manage packages that will be used globally across the Windows operating system.

🪠 %shell> choco install --yes git python vscode

🪠 ➡️ add [custom settings](settings-user.json) into **User-scope settings.json** in VSCode settings (⌨️ Ctrl+,)

| Type                 | Name   | Version                      | Reference                                                                 |
| -------------------- | ------ | ---------------------------- | ------------------------------------------------------------------------- |
| SCM                  | git    | **Latest Stable;** >= 2.46.0 | [Install Git](https://community.chocolatey.org/packages/git)              |
| Programming Language | python | **Latest Stable;** >= 3.12.5 | [Install Python](https://community.chocolatey.org/packages/python/3.12.5) |
| IDE                  | code   | **Latest Stable;** >= 1.92.2 | [Install VS Code](https://community.chocolatey.org/packages/vscode)       |

---

#### In Ubuntu
🪠 To run "init_in_bash.sh" automatically calls "init_in_fish.fish" in "_initialization" directory

- ➡️ [Initialization script in **Bash shell**](prototypes/_initialization/init_in_bash.sh)

- ➡️ [Initialization script in **Fish shell**](prototypes/_initialization/init_in_fish.fish)

| Type  | Name            | Version                         | Reference                                                      |
| ----- | --------------- | ------------------------------- | -------------------------------------------------------------- |
| OS    | \[WSL2\] Ubuntu | **Latest Stable;** >= 24.04 LTS | by .iso file or Microsoft Store                                |
| Shell | fish            | **Latest Stable;** >= 3.7.1     | [Install Fish shell](https://github.com/fish-shell/fish-shell) |
| SCM   | git             | **Latest Stable;** >= 2.43.0    | (default package)                                              |


- 🧰 Troubleshooting
  - If you need to shrink a drive's capacity for Ubuntu installation and the amount of space you can shrink is less than the free space, use the tool [MiniTools partition wizard](https://www.partitionwizard.com/) in order to do partitions.

##### Packages Managed by "apt" command in "fish" shell

| Type                        | Name  | Version                       | Reference                                             |
| --------------------------- | ----- | ----------------------------- | ----------------------------------------------------- |
| Python Isolated Environment | pipx  | **Latest Stable;** >= 1.4.3   | [Install pipx](https://pipx.pypa.io/stable/#on-linux) |
| Python Version Manager      | pyenv | **Latest Stable;** >= 2.41.10 | [Install pyenv](https://github.com/pyenv/pyenv)       |

##### Packages Managed by "pipx" command in "fish" shell

| Type                   | Name   | Version                     | Reference                                       |
| ---------------------- | ------ | --------------------------- | ----------------------------------------------- |
| Python Package Manager | poetry | **Latest Stable;** >= 1.8.3 | [Install pyenv](https://github.com/pyenv/pyenv) |
| C/C++ Package Manager  | conan  | **Latest Stable;** >= 2.7.0 | [Install conan](https://github.com/pyenv/pyenv) |
- 📝 After to use command "poetry install", run the VSCode command in each project
  ```bash
    # VSCode command
    > Python: Select Interpreter # set as python executable path in installed virtual environment
  ```


###### Update commands
🪠 (fish shell)
```bash
# fish shell
sudo apt update -y
pyenv update
sudo apt upgrade -y
```

### 1. VSCode Extension Installation

- Install recommended extension of VScode (by .vscode/extensions.json)

---

### 2. Other settings

Other settings is already written to integrate with VSCode.: **Intellisense**, **Toolchain**, clang-format

- .vscode/c_cpp_properties.json
- .vscode/settings.json
- .vscode/tasks.json
- CMakeLists.txt
- conanfile.py


---

### 4. \[Optional\] Device settings

If you want to use USB device specifically Camera, you must set following options.

#### USB camera

1. First, check connected USB Camera Device is in Device Manager - Camera

2. Windows Settings
   - Privacy & security
     - Camera
       - ☑️ Camera access
       - ☑️ Let apps access your camera
         - ☑️ Let desktop apps access your camera

- FAQ
  - ❓ When using multiple cameras, there were cases where opencv-python did not recognize two cameras on the same port hub.
     > cv2.VideoCapture(index).isOpend() is True, but frame returned by read() is False.

---

## Order of Tasks for using C++

1. Configure my cpp sources

   ```cmake
   # 🛍️ e.g. CMake configuration for cpp_study target
   # ➡️ whenever structure of directory is changed, use this.
   add_executable(cpp_study
       src/cpp_study.cpp
       src/main.cpp
       <new cpp source1>
       <new cpp source2> # ...
   )
   ```

2. Install dependencies and set the build type to Debug:

   ```bash
   # shell command
   # ➡️ whenever conanfile.py is changed, use this.
   conan install . -s build_type=Debug --build=missing
   ```

3. CMake Configure Preset; configure preset:

   ```bash
   # shell command
   # ➡️ whenever CMakeLists is changed, use this.
   cmake --preset conan-debug
   ```

4. CMake Build; build project:

   ```bash
   # shell command
   # ➡️ whenever (source | include) file is changed, use this.
   cmake --build --preset conan-debug
   ```

5. Run the created executable:

   ```bash
   # shell command
   ./build/Debug/cpp_study
   ```

## Changelog

- 📅 2024-08-26 21:50:50
  - The settings environment is changed from WSL Ubuntu settings to Windows 11 because WSL2 Ubuntu not supports well USB camera drivers.
     > I have tried building with a custom kernel and followed all the steps for [USB device sharing](https://learn.microsoft.com/en-us/windows/wsl/connect-usb) as recommended by Microsoft. However, I still could not access video devices from OpenCV in WSL2 (/dev/video* are None).
- 📅 2024-08-29 00:19:46
  - I decided to re-use WSL2 Ubuntu in home, and use Ubuntu OS (not WSL) in outside.
  - Many C++ tools and libraries seem to be tailored to the unix-lke operating system.
    - Even tools that are known to work on Windows, when applied with custom options, result in many build errors for the libraries, and the work to resolve them is also limited and cumbersome.
    - If you want to use USB Device Camera in Opencv in local, work it outside such as at company.

## legacy

### Packages Managed by "pixi" command

   ~~Package Manager for Cross-platform~~ | ~~pixi~~   | ~~**Latest Stable;** >= 0.28.1~~                 | ~~[Install pixi](https://pixi.sh/)~~                                                                                  |

- pixi is a very new tool, and if I only going to use Unix-like OS, I probably better off using another tool.

   🗝️ This tool is used to install and manage packages for project-specific isolated environment configurations.
   ⚠️ Do Not install these as global package

   There are common packages that you may need to install for development on certain systems.

   | Type                              | Name   | Version                     | Reference                                          |
   | --------------------------------- | ------ | --------------------------- | -------------------------------------------------- |
   | Package Manager for C++           | conan  | **Latest Stable;** >= 2.6.0 | [Install Conan 2](https://pypi.org/project/conan/) |
   | Python interpreter                | python | **\<Custom Version\>**      |                                                    |
   | Tools for install python packages | pip    | **Latest Stable;** >= 2.6.0 | [Install pip](https://pypi.org/project/conan/)     |

- pip is only used when [Python Interactive window](https://code.visualstudio.com/docs/python/jupyter-support-py) with [Jupyter extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter) in VS code for development. not for Release

   🪠 Refer to each project prototypes for package details ➡️

   ---

## 📰 Doing
### Installation in order to build with clang

```bash
# shell command
sudo apt install -y cmake build-essential llvm clang libc++-dev libc++abi-dev \
  libva-dev libvdpau-dev libx11-xcb-dev libfontenc-dev libice-dev \
  libsm-dev libxaw7-dev libxcomposite-dev libxcursor-dev libxdamage-dev \
  libxext-dev libxfixes-dev libxi-dev libxinerama-dev libxkbfile-dev \
  libxmu-dev libxmuu-dev libxpm-dev libxrandr-dev libxrender-dev \
  libxres-dev libxss-dev libxt-dev libxtst-dev libxv-dev \
  libxxf86vm-dev libxcb-glx0-dev libxcb-render0-dev libxcb-render-util0-dev libxcb-xkb-dev \
  libxcb-icccm4-dev libxcb-image0-dev libxcb-keysyms1-dev libxcb-randr0-dev libxcb-shape0-dev \
  libxcb-sync-dev libxcb-xfixes0-dev libxcb-xinerama0-dev libxcb-dri3-dev uuid-dev \
  libxcb-cursor-dev libxcb-dri2-0-dev libxcb-dri3-dev libxcb-present-dev libxcb-composite0-dev \
  libxcb-ewmh-dev libxcb-res0-dev libxcb-util-dev libxcb-util0-dev \
  pkg-config ccache
```

📰🚨 Issue tracking
  - https://github.com/conan-io/conan/issues/16905
  - https://github.com/conan-io/conan-center-index/issues/25075

