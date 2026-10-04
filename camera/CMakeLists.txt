cmake_minimum_required(VERSION 3.10)
project(daheng_camera LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# OpenCV: find_package 找不到时降级为只用头文件 + pkg-config
find_package(OpenCV REQUIRED)

#日志
find_package(spdlog REQUIRED)


# 大恒 Galaxy SDK 头文件(include/sdk/GxIAPI.h、include/sdk/DxImageProc.h)
set(DAHENG_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

# 大恒 SDK 库。已安装到 /usr/lib 时下面的 find_library 会自动找到;
# 若没有安装,改成 SDK 实际路径,例如:
#   /home/hl/Downloads/Galaxy_Linux-x86_Gige-U3_32bits-64bits_xxx/Galaxy_camera/lib/x86_64
set(DAHENG_SDK_LIB_DIR "" CACHE PATH "大恒 Galaxy SDK 库目录")

find_library(DAHENG_LIB
    NAMES gxiapi
    HINTS ${DAHENG_SDK_LIB_DIR}
    PATHS /usr/lib /usr/local/lib
)

if(NOT DAHENG_LIB)
    message(FATAL_ERROR
        "找不到大恒 SDK 库 libgxiapi.so。"
        "请用 -DDAHENG_SDK_LIB_DIR=<SDK>/lib/x86_64 指定目录。")
endif()

message(STATUS "大恒 SDK 库: ${DAHENG_LIB}")

add_executable(daheng daheng.cc)

target_include_directories(daheng PRIVATE ${DAHENG_INCLUDE_DIR})

# DxRaw8toRGB24Ex 在新版本 SDK 中已经并入 libgxiapi.so;
# 老版本需要额外链接 libDxImageProc.so,取消下面这行注释即可。
target_link_libraries(daheng PRIVATE ${DAHENG_LIB} ${OpenCV_LIBS} spdlog::spdlog)

# 运行时能找到 libgxiapi.so(如果 SDK 只在非标准路径)
get_filename_component(DAHENG_LIB_DIR ${DAHENG_LIB} DIRECTORY)
target_link_options(daheng PRIVATE "-Wl,-rpath,${DAHENG_LIB_DIR}")