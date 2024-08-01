cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(hand_infer)

find_package(PCL REQUIRED)
find_package (OpenCV REQUIRED)

include_directories(${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS} ${OpenCV_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})


add_executable (hand_infer cpp/yolo_pointcloud.cpp)
target_link_libraries (hand_infer ${PCL_LIBRARIES} ${OpenCV_LIBS})

