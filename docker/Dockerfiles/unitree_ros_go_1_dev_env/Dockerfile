# from https://hub.docker.com/_/microsoft-vscode-devcontainers
FROM mcr.microsoft.com/devcontainers/cpp:ubuntu-18.04
LABEL maintainer="wbfw109@gmail.com"
LABEL source="https://github.com/wbfw109/study-core/tree/dev/docker/Dockerfiles/unitree_ros_go_1_dev_env"

# settings
SHELL ["/bin/bash", "-c"]
WORKDIR /opt

# install ROS 1 Melodic
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN apt install -y curl
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
RUN apt update

RUN apt install -y ros-melodic-desktop-full
RUN apt search ros-melodic
RUN echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
RUN source /opt/ros/melodic/setup.bash && \
  apt install -y python-rosdep python-rosinstall python-rosinstall-generator python-wstool build-essential && \
  rosdep init && rosdep update

## download sources
# curl -L: in case there is a redirect found.
RUN mkdir unitree/catkin_ws/src -p
ENV PWD=/opt/unitree/catkin_ws/src
WORKDIR ${PWD}

RUN curl --silent --show-error --location -o file.zip https://github.com/unitreerobotics/unitree_ros/archive/master.zip && \
    unzip file.zip && rm file.zip && mv unitree_ros-master unitree_ros
RUN curl --silent --show-error --location -o file.zip https://github.com/unitreerobotics/unitree_legged_sdk/archive/go1.zip && \
    unzip file.zip && rm file.zip && mv unitree_legged_sdk-go1 unitree_legged_sdk
RUN curl --silent --show-error --location -o file.zip https://github.com/unitreerobotics/unitree_ros_to_real/archive/master.zip && \
    unzip file.zip && rm file.zip && \
    mv unitree_ros_to_real-master/unitree_legged_real unitree_legged_real && \
    mv unitree_ros_to_real-master/unitree_legged_msgs unitree_legged_msgs && \
    rm -r unitree_ros_to_real-master

# pro-process for unitree_ros repository
RUN apt install -y ros-melodic-controller-interface ros-melodic-gazebo-ros-control ros-melodic-joint-state-controller ros-melodic-effort-controllers ros-melodic-joint-trajectory-controller
RUN sed -i "s#/home/unitree/catkin_ws/src#$PWD#g" unitree_ros/unitree_gazebo/worlds/stairs.world

### Build
# build <unitree_legged_sdk>
RUN mkdir unitree_legged_sdk/build
ENV PWD=/opt/unitree/catkin_ws/src/unitree_legged_sdk/build
WORKDIR ${PWD}
RUN source /opt/ros/melodic/setup.bash && cmake ../ && make
RUN echo "source $PWD/devel/setup.bash" >> ~/.bashrc

# catkin_make remainders
ENV PWD=/opt/unitree/catkin_ws
WORKDIR ${PWD}
RUN source /opt/ros/melodic/setup.bash && catkin_make
RUN echo "source $PWD/devel/setup.bash" >> ~/.bashrc

### pro-process for message "Could not find the GUI, install the 'joint_state_publisher_gui' package" in headless environment
RUN apt install -y ros-melodic-joint-state-publisher-gui

## ➡️ test commands! from 🔗 https://github.com/unitreerobotics/unitree_ros#detail-of-packages

# Entrypoint
ENTRYPOINT ["/bin/bash"]

## build this Dockerfile for multi-platforms and push remote.
# %shell> docker buildx build --platform linux/amd64,linux/arm64 -t wbfw109/unitree_ros_go_1_dev_env:1 -t wbfw109/unitree_ros_go_1_dev_env:latest --push .
# ❔ to build linux/arm64 not works well in WSL 2 Ubutnu 22.04 (linux/amd64) 😭 QEMU: UNCAUGHT TARGET SIGNAL 7 (BUS ERROR) - CORE DUMPED
