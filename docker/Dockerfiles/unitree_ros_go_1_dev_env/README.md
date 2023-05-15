# README

- [README](#readme)
  - [1. How to make the Development environment for Unitree ROS 1 (Melodic)?](#1-how-to-make-the-development-environment-for-unitree-ros-1-melodic)
    - [1.1. Use pre-made Docker image](#11-use-pre-made-docker-image)
    - [1.2. Build Docker image yourself](#12-build-docker-image-yourself)
      - [1.2.1 Create the Docker image from Dockerfile](#121-create-the-docker-image-from-dockerfile)
      - [1.2.2. Install manually in a container](#122-install-manually-in-a-container)
  - [2. ❔Questions](#2-questions)
    - [2.1. How to run commands?](#21-how-to-run-commands)
    - [2.2. How to display GUI by commands (in headless environment)?](#22-how-to-display-gui-by-commands-in-headless-environment)

🎉 Welcome to Unitree ROS 1 (Melodic) Development environment

➡️ [Test commands](https://github.com/unitreerobotics/unitree_ros#detail-of-packages)

## 1. How to make the Development environment for Unitree ROS 1 (Melodic)?

### 1.1. Use [pre-made Docker image](https://hub.docker.com/repository/docker/wbfw109/unitree_ros_go_1_dev_env/general)

⭕ **(Recommended)**

use `>Dev Containers: Reopen in Container` command in VS Code with `.devcontainer/devcontainer.json`

- ```json
  {
    "name": "Unitree ROS 1 (Melodic) Development environment",
    "image": "wbfw109/unitree_ros_go_1_dev_env:latest",
    "remoteUser": "root"
  }
  ```

- in this case, it is already set for remote user as **root**, so you do not have to run `su --login`.

### 1.2. Build Docker image yourself

📝 If your computer architecture is different with **linux/amd64 (pre-made image's architecture**), you have to the build image yourself.

#### 1.2.1 Create the Docker image from Dockerfile

⭕ **(Recommended)**

in shell, `docker build --tag <company>/<image_name>:<version> <Dockerfile_path>`

#### 1.2.2. Install manually in a container

for example, in VSCode

- use VS Code and run `>Dev Containers: Reopen in Container` command in VS Code with `.devcontainer/devcontainer.json`

  - ```json
    {
      "name": "Ubuntu",
      "image": "mcr.microsoft.com/devcontainers/cpp:ubuntu-18.04"
    }
    ```

- run [setting_unitree_go_1.sh](setting_unitree_go_1.sh)
  - It is installed as normal **user**, not root.

---

## 2. ❔Questions

### 2.1. How to run commands?

This Docker image is written as **root user**.
so, you need to change user by `su --login`.

### 2.2. How to display GUI by commands (in headless environment)?

- [use VS Code command](https://code.visualstudio.com/docs/devcontainers/containers) `>Dev Containers: Reopen in Container`.
  - [It automatically connext to your display if your OS is Windows and you are using WSL2. (📝 I've not tested in other OS.)](#11-use-pre-made-docker-image)
- or use communication protocol that specifies the communication between a display server and its clients like Wayland, X11.
