#!/bin/bash
###### fail....................... 📅 2024-09-18 22:18:18

# Automated script: Build Mesa and enable Microsoft experimental driver

# 2. Install essential build tools and related dependencies
sudo apt install -y build-essential meson ninja-build autoconf automake libtool

# 3. Install X11, Wayland, and related graphical libraries
sudo apt install -y libx11-dev libxext-dev libwayland-dev libxcb-glx0-dev libxshmfence-dev libwayland-egl-backend-dev

# 4. Install compression, expat, and PCI access libraries
sudo apt install -y zlib1g-dev libexpat1-dev libpciaccess-dev

# 5. Ensure latest LLVM and Vulkan packages from the official Ubuntu 24 repositories
echo "Installing LLVM 18 and SPIR-V tools from official Ubuntu 24 repositories..."

# Update package lists
sudo apt update

# Install LLVM 18 and SPIR-V tools from Ubuntu 24 repositories
sudo apt install llvm-18 llvm-18-dev libllvmspirvlib-18-dev libvulkan-dev libvulkan1 spirv-tools

# Install the appropriate version of libclc
sudo apt install -y libclc-18-dev libclc-15-dev  # 15버전도 설치

# Check if libclc is correctly installed
dpkg -L libclc-18-dev
dpkg -L libclc-15-dev  # 15버전 확인

# Set OpenCL environment variable
export OPENCL_HOME=/usr/lib/llvm-18/lib/clang/18/include

# 6. Install Clang development packages for LLVM 18
sudo apt install -y clang-18 libclang-18-dev

# 7. Set llvm-config to use LLVM 18
export LLVM_CONFIG=/usr/bin/llvm-config-18

# 8. Set pkg-config path to use LLVM 18
export PKG_CONFIG_PATH=/usr/lib/llvm-18/lib/pkgconfig:$PKG_CONFIG_PATH

# 9. Verify if pkg-config recognizes llvmspirvlib
echo "Checking if pkg-config recognizes llvmspirvlib..."
pkg-config --modversion llvmspirvlib
if [ $? -ne 0 ]; then
    echo "llvmspirvlib not found. Building llvm-spirv from source..."
    
    # 10. Clone and build llvm-spirv from source
    mkdir -p ~/src
    cd ~/src

    # Clone the SPIRV-LLVM-Translator repository
    git clone https://github.com/KhronosGroup/SPIRV-LLVM-Translator.git
    cd SPIRV-LLVM-Translator

    # Create a build directory
    mkdir build
    cd build

    # Configure the build with CMake (ensure LLVM 18 is being used)
    cmake .. -DLLVM_DIR=$(llvm-config --cmakedir)

    # Build and install llvm-spirv
    make
    sudo make install
else
    echo "llvmspirvlib found!"
fi

# 11. Install graphics-related libraries (GL, VDPAU, etc.)
sudo apt install -y libglvnd-dev libvdpau-dev

# 12. Install remaining dependencies (Bison, Flex, Python, etc.)
sudo apt install -y bison flex python3-mako python3-pip python3-setuptools

# Install Python PLY module required for GRL kernels
pip3 install ply

# 13. Install remaining dependencies (Bison, Flex, Python, etc.)
sudo apt install -y bison flex python3-mako python3-pip python3-setuptools

# 14. Clone and build the latest version of libdrm (>=2.4.121)
echo "Cloning and building libdrm (>=2.4.121)..."
mkdir -p ~/src/drm
cd ~/src/drm

# Check if git repository is already cloned
if [ ! -d ".git" ]; then
  git clone https://gitlab.freedesktop.org/mesa/drm.git .
fi

mkdir -p build
cd build

# Reconfigure the Meson build if it is already set up
meson setup --reconfigure ..

ninja
sudo ninja install

# Set pkg-config path for the installed libraries
export PKG_CONFIG_PATH=/usr/local/lib/x86_64-linux-gnu/pkgconfig:/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH

# 15. Clone Mesa source code and build it (working in the home directory)
echo "Cloning Mesa repository into ~/src/mesa..."
mkdir -p ~/src/mesa
cd ~/src/mesa

if [ ! -d ".git" ]; then
  echo "Cloning Mesa source code..."
  git clone https://gitlab.freedesktop.org/mesa/mesa.git .
else
  echo "Mesa source directory already exists, updating source code..."
  git pull
fi

# 16. Set up the build directory and prepare the build
echo "Setting up build directory in ~/src/mesa/build..."
meson setup build/ --prefix=/usr/local -D vulkan-drivers=amd,intel,intel_hasvk,swrast,virtio,nouveau,microsoft-experimental -D intel-clc=enabled  # intel-clc 옵션 추가

# 17. Build Mesa with the Microsoft experimental driver
echo "Building Mesa with Microsoft experimental driver enabled in ~/src/mesa/build..."
ninja -C build/

# 18. Install the built libraries to /usr/local
echo "Installing Mesa to /usr/local..."
sudo ninja -C build/ install

# 19. Set up the Microsoft experimental driver and check Vulkan setup
echo "Setting up Microsoft experimental driver for Vulkan..."

# Set environment variables to use the Microsoft experimental driver
export VK_ICD_FILENAMES=/usr/local/share/vulkan/icd.d/dzn_icd.x86_64.json

# Check the Vulkan setup and GPU information
echo "Running vulkaninfo to check setup..."
vulkaninfo | grep -i "GPU"

echo "Mesa build and installation with Microsoft experimental driver complete in /usr/local."


