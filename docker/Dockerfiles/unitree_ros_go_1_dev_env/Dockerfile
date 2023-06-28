# from https://hub.docker.com/_/microsoft-vscode-devcontainers
FROM mcr.microsoft.com/devcontainers/cpp:ubuntu-18.04
LABEL maintainer="wbfw109@gmail.com"
LABEL source="https://github.com/wbfw109/study-core/tree/dev/docker/Dockerfiles/unitree_ros_go_1_dev_env"

# settings
WORKDIR /home
RUN mkdir -p installation
COPY ./* installation/
RUN chmod +x installation/setting_unitree_go_1.sh

# instsall
RUN installation/setting_unitree_go_1.sh


# Entrypoint
ENTRYPOINT ["/bin/bash"]

## build this Dockerfile for multi-platforms and push remote.
# %shell> docker buildx build --platform linux/amd64,linux/arm64 -t wbfw109/unitree_ros_go_1_dev_env:1 -t wbfw109/unitree_ros_go_1_dev_env:latest --push .