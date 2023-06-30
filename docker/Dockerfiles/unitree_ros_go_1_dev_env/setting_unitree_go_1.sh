### ROS project Commands
## from
# https://github.com/unitreerobotics/unitree_ros
# - https://wiki.ros.org/Installation/Ubuntu - Melodic
# - https://github.com/unitreerobotics/unitree_ros_to_real
#   - https://github.com/unitreerobotics/unitree_legged_sdk

# install ROS 1 Melodic
working_dir="$PWD"
echo "$working_dir"

sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
sudo apt install -y curl # if you haven't already installed curl
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
sudo apt update
sudo apt install -y ros-melodic-desktop-full
apt search ros-melodic
echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
source ~/.bashrc
sudo apt install -y python-rosdep python-rosinstall python-rosinstall-generator python-wstool build-essential
sudo rosdep init
rosdep update

## download sources
# curl -L: in case there is a redirect found.
mkdir unitree/catkin_ws/src -p && cd unitree/catkin_ws/src

curl --silent --show-error --location -o file.zip https://github.com/unitreerobotics/unitree_ros/archive/master.zip && \
    unzip file.zip && rm file.zip && mv unitree_ros-master unitree_ros
curl --silent --show-error --location -o file.zip https://github.com/unitreerobotics/unitree_legged_sdk/archive/go1.zip && \
    unzip file.zip && rm file.zip && mv unitree_legged_sdk-go1 unitree_legged_sdk
curl --silent --show-error --location -o file.zip http://github.com/unitreerobotics/unitree_ros_to_real/archive/master.zip && \
    unzip file.zip && rm file.zip &&\
    mv unitree_ros_to_real-master/unitree_legged_real unitree_ros_to_real &&\
    mv unitree_ros_to_real-master/unitree_legged_msgs unitree_legged_msgs &&\
    rm -r unitree_ros_to_real-master

# pro-process for unitree_ros repository
sudo apt install -y ros-melodic-controller-interface  ros-melodic-gazebo-ros-control ros-melodic-joint-state-controller ros-melodic-effort-controllers ros-melodic-joint-trajectory-controller
sed -i "s#/home/unitree/catkin_ws/src#$PWD#g" unitree_ros/unitree_gazebo/worlds/stairs.world

### Build
# build <unitree_legged_sdk>
cd unitree_legged_sdk && mkdir build && cd build && cmake ../ && make
echo "source $PWD/devel/setup.bash" >> ~/.bashrc
source ~/.bashrc

# catkin_make remainders
cd ../../../ && catkin_make
echo "source $PWD/devel/setup.bash" >> ~/.bashrc
source ~/.bashrc

### pro-process for message "Could not find the GUI, install the 'joint_state_publisher_gui' package" in headless environment
sudo apt install -y ros-melodic-joint-state-publisher-gui

## ➡️ test commands! from 🔗 https://github.com/unitreerobotics/unitree_ros#detail-of-packages



### deprecated (no longer used)
# curl https://github.com/unitreerobotics/unitree_legged_sdk/archive/refs/tags/v3.8.0.zip -Ss -L -o file.zip && \
#     unzip file.zip && rm file.zip && mv unitree_legged_sdk-3.8.0 unitree_legged_sdk

### Note that If you use WSL2, add Google public domain (8.8.8.8 and 8.8.4.4) into /etc/resolv.conf.
# otherwise, when some `apt install` packages, you may get Error mesage `Failed to fetch ... Connection failed ... E: Unable to fetch some archives, maybe run apt-get update or try with --fix-missing?`.
