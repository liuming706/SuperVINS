#!/bin/bash

RCol="\e[0m"
Red="\e[0;31m"
Gre="\e[0;32m"
Yel="\e[0;33m"
Blu="\e[0;34m"

T800_DIR=/home/ubt/workspace/t800
VNAV_DIR=$T800_DIR/vnav
SUPERVISOR_DIR=$T800_DIR/supervisor
TensorRT_ROOT=/home/ubt/workspace/3rdparty/release/TensorRT-8.6.1.6
Torch_DIR=/home/ubt/workspace/3rdparty/release/libtorch/share/cmake/Torch
BUILD_PATH=$(pwd)/build_x86
INSTALL_PATH=$(pwd)/install_x86 

export CPATH=/usr/local/cuda/targets/x86_64-linux/include:$CPATH
export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda/bin:$PATH
export CMAKE_PREFIX_PATH=/opt/nvidia/vpi3:$CMAKE_PREFIX_PATH

DIR=$($dirname; pwd)
echo -e "${Blu}T800_DIR:$Yel $T800_DIR $RCol"
echo -e "${Blu}VNAV_DIR:$Yel $VNAV_DIR $RCol"
echo -e "${Blu}SUPERVISOR_DIR:$Yel $SUPERVISOR_DIR $RCol"

source /opt/ros/noetic/setup.bash
mkdir -p $T800_DIR
mkdir -p $VNAV_DIR
mkdir -p $SUPERVISOR_DIR
rm -rf build_x86/voxblox/Block.pb.h
rm -rf build_x86/voxblox/Block.pb.cc
    # --symlink-install \
    # --packages-up-to \
    #   vnav_config_models \
    #   vnav_config_t800 \
    #   vnav_tf_processor \
    #   apriltag_detector \
    #   cv_task_msgs \
    #   vslam \
    #   vnav_landmark_pub \

    # --packages-up-to \
    #   apriltag_detector \
    #   vnav_config_models \
    #   vnav_tf_processor \
    #   vnav_config_webots \
    #   vslam \
    #   vnav_landmark_pub \
    # -DCUDAToolkit_ROOT=/usr/local/cuda \
echo $HOME
catkin_make \
    --cmake-args \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=Yes \
    -DTensorRT_ROOT=$TensorRT_ROOT \
    -DTorch_DIR=$Torch_DIR \
    -DONNXRUNTIME_ROOTDIR=$HOME/workspace/3rdparty/release/onnxruntime-linux-x64-gpu-1.16.3 \
    -DCERES_DIR=$HOME/workspace/3rdparty/release/ceres-solver-2.1.0/install
# colcon build  \
#     --build-base $BUILD_PATH \
#     --install-base $INSTALL_PATH \
#     --symlink-install \
#     --packages-up-to \
#       vslam  \
#     --mixin  \
#       compile-commands \
#       shared \
#       rel-with-deb-info \
#     --cmake-force-configure \
#     --cmake-args \
#         -DTensorRT_ROOT=$TensorRT_ROOT \
#         -DTorch_DIR=$Torch_DIR \
#       $1 $2 $3 $4 $5 $6 $7

# 拷贝系统 apriltag 库
# rsync  -azqP algorithm/vnav_landmark_pub/apriltag_install.zip $VNAV_DIR/
# unzip  -o $VNAV_DIR/apriltag_install.zip -d  $VNAV_DIR
# rsync -azqP $VNAV_DIR/apriltag_install/*  $VNAV_DIR

echo -e "${Yel}compile end"
