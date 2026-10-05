cmake_minimum_required(VERSION 3.10)
project(armor_recognize LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV REQUIRED)
find_package(spdlog REQUIRED)

# 大恒 Galaxy SDK 头文件复用 camera/include/sdk(GxIAPI.h、DxImageProc.h)
set(DAHENG_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../camera/include)
if(NOT EXISTS ${DAHENG_INCLUDE_DIR}/sdk/GxIAPI.h)
    message(FATAL_ERROR
        "找不到大恒 SDK 头文件:${DAHENG_INCLUDE_DIR}/sdk/GxIAPI.h")
endif()

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

add_executable(recognize recognize.cc)

target_include_directories(recognize PRIVATE ${DAHENG_INCLUDE_DIR})
target_link_libraries(recognize PRIVATE ${DAHENG_LIB} ${OpenCV_LIBS} spdlog::spdlog)

# 运行时能找到 libgxiapi.so(如果 SDK 只在非标准路径)
get_filename_component(DAHENG_LIB_DIR ${DAHENG_LIB} DIRECTORY)
target_link_options(recognize PRIVATE "-Wl,-rpath,${DAHENG_LIB_DIR}")