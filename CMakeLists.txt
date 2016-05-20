# CMake 最低版本号要求
cmake_minimum_required(VERSION 2.8.7)

# 项目信息
project(struck)

# include路径
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/eigen)
include_directories(/usr/local/include)

# 设置可执行文件生成路径
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

# 生成debug版本
SET(CMAKE_BUILD_TYPE "release")
if (CMAKE_BUILD_TYPE STREQUAL debug)
    add_definitions(-D_DEBUG)
endif ()
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb -std=c++11")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall -std=c++11")

# 查找当前目录下的所有源文件
# 并将名称保存到 DIR_LIB_SRCS 变量
aux_source_directory(./src DIR_SRCS)

# 指定生成目标
add_executable(struck ${DIR_SRCS})

set(OpenCV_DIR /usr/local/opencv-2.4.11/build)
message(${OpenCV_DIR})
set(OpenCV_LIBS -lopencv_calib3d -lopencv_contrib -lopencv_core 
-lopencv_features2d -lopencv_flann -lopencv_gpu -lopencv_highgui 
-lopencv_imgproc -lopencv_legacy -lopencv_ml -lopencv_nonfree 
-lopencv_objdetect -lopencv_ocl -lopencv_photo -lopencv_stitching 
-lopencv_superres -lopencv_ts -lopencv_video -lopencv_videostab 
-lopencv_viz -lcufft -lnpps -lnppi -lnppc -lcudart -ltbb -lXext 
-lX11 -lICE -lSM -lGL -lGLU -lrt -lpthread -lm -ldl)
#message(${OpenCV_LIBS})

# 添加链接库目录
#find_package(OpenCV REQUIRED) #this sentence get wrong, so i set OpenCV_LIBS manually
target_link_libraries(struck ${OpenCV_LIBS})
