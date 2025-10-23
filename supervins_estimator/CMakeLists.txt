cmake_minimum_required(VERSION 2.8.3)
project(supervins)
# cmake_policy(SET CMP0074 NEW)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_CXX_STANDARD 14)
# -DEIGEN_USE_MKL_ALL")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(
  catkin REQUIRED
  COMPONENTS roscpp
             std_msgs
             geometry_msgs
             nav_msgs
             tf
             cv_bridge
             camera_models
             image_transport)

find_package(OpenCV REQUIRED)

if(NOT CERES_DIR)
  message(WARNING "CERES_DIR not set, using default")
  find_package(Ceres REQUIRED)
else()
  find_package(Ceres REQUIRED PATHS ${CERES_DIR})
  message(STATUS "Using CERES_DIR: ${CERES_DIR}")
endif()

# set(ONNXRUNTIME_ROOTDIR
# "/home/ubt/workspace/3rdparty/release/onnxruntime-linux-x64-gpu-1.16.3")
include_directories("${ONNXRUNTIME_ROOTDIR}/include")
link_directories("${ONNXRUNTIME_ROOTDIR}/lib")

include_directories(${PROJECT_SOURCE_DIR}/featureTracker/ort_include/)
include_directories(${PROJECT_SOURCE_DIR}/src/estimator/)
include_directories(${catkin_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS})

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3)
include_directories(${catkin_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})

catkin_package()

add_library(
  supervins_lib
  src/estimator/parameters.cpp
  src/estimator/estimator.cpp
  src/estimator/feature_manager.cpp
  src/factor/pose_local_parameterization.cpp
  src/factor/projectionTwoFrameOneCamFactor.cpp
  src/factor/projectionTwoFrameTwoCamFactor.cpp
  src/factor/projectionOneFrameTwoCamFactor.cpp
  src/factor/marginalization_factor.cpp
  src/utility/utility.cpp
  src/utility/visualization.cpp
  src/utility/CameraPoseVisualization.cpp
  src/initial/solve_5pts.cpp
  src/initial/initial_aligment.cpp
  src/initial/initial_sfm.cpp
  src/initial/initial_ex_rotation.cpp
  src/featureTracker/feature_tracker.cpp
  src/featureTracker/feature_tracker_dpl.cpp
  src/featureTracker/extractor_matcher_dpl.cpp
  src/featureTracker/transform_dpl.cpp)
target_link_libraries(supervins_lib PUBLIC ${catkin_LIBRARIES} ${OpenCV_LIBS}
                                           Ceres::ceres onnxruntime)
# target_include_directories(supervins_lib PRIVATE ${CERES_INCLUDE_DIRS})
message("CERES_INCLUDE_DIRS: " ${CERES_INCLUDE_DIRS})
# 查找 CUDA
find_package(CUDA REQUIRED)
if(CUDA_FOUND)
  message(STATUS "CUDA found: ${CUDA_VERSION}")
  target_compile_definitions(supervins_lib PRIVATE USE_CUDA)
  target_include_directories(supervins_lib PRIVATE ${CUDA_INCLUDE_DIRS})
  target_link_libraries(supervins_lib PRIVATE ${CUDA_LIBRARIES})
endif()
get_filename_component(PROJECT_SOURCE_DIR_ABS "${CMAKE_SOURCE_DIR}" ABSOLUTE)
add_executable(supervins_node src/supervins_main.cpp)
target_compile_definitions(
  supervins_node PRIVATE PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR_ABS}")
target_link_libraries(supervins_node supervins_lib)

# add_executable(kitti_odom_test src/KITTIOdomTest.cpp )
# target_link_libraries(kitti_odom_test vins_lib -pthread)

# add_executable(kitti_gps_test src/KITTIGPSTest.cpp)
# target_link_libraries(kitti_gps_test vins_lib -pthread)
