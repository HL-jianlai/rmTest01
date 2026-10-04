// 大恒(Galaxy)相机 USB3 相机取流 Demo
// 由 1.cc ~ 8.cc 合并而成,按学习顺序逐级累积:
//   1. GXInitLib / GXCloseLib      —— 初始化库
//   2. EnumerateDevices            —— 枚举设备
//   3. GetErrorString              —— 错误信息
//   4. FindDeviceIndexBySerial     —— 按序列号定位设备
//   5. AddExposureTime             —— 设置参数(曝光)
//   6. AddGain / AddGamma          —— 设置参数(增益 / 伽马)
//   7. 打开设备 + 配置 + 取流 + 显示 + FPS
//   8. 循环取帧 + QB 还帧
//
// 编译:见同目录 CMakeLists.txt
// 用法:./daheng [相机序列号]   省略序列号则自动使用第 1 台 USB3 相机

#include <sdk/GxIAPI.h>
#include <sdk/DxImageProc.h> // 图像处理接口:DxRaw8toRGB24Ex 等

#include <opencv2/opencv.hpp>

#include <chrono>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include<spdlog/spdlog.h>


// ------------------------------------------------------------------
// 设备信息
// ------------------------------------------------------------------
struct DeviceInfo {
    std::string serial;
    std::string model;
};

// ------------------------------------------------------------------
// 取 SDK 的详细错误文本(1.cc 只打印状态码,这里补上原因)
// GXGetLastError 要调两次:第一次只取缓冲区长度,第二次才取内容
// ------------------------------------------------------------------
std::string GetErrorString(GX_STATUS emErrorStatus) {
    char *error_info = nullptr;
    size_t size = 0;
    GX_STATUS emStatus = GXGetLastError(&emErrorStatus, nullptr, &size); // 第一次:只取长度
    if (emStatus != GX_STATUS_SUCCESS) {
        return "<Error when calling GXGetLastError>";
    }
    error_info = new char[size];
    emStatus = GXGetLastError(&emErrorStatus, error_info, &size);        // 第二次:取内容
    std::string error_string = error_info != nullptr ? error_info : "";
    delete[] error_info;
    return emStatus == GX_STATUS_SUCCESS ? error_string : "<Error when calling GXGetLastError>";
}

// ------------------------------------------------------------------
// 枚举设备(只收 USB3 相机,GigE 相机的序列号在 stGigEDevInfo 里)
// ------------------------------------------------------------------
std::vector<DeviceInfo> EnumerateDevices() {
    uint32_t device_num = 0;
    // 第二个参数是枚举超时(ms):GigE 相机响应慢,官方建议至少 1000
    GX_STATUS emStatus = GXUpdateAllDeviceList(&device_num, 1000);
    if (emStatus != GX_STATUS_SUCCESS || device_num == 0) {
    //    std::cout << "枚举失败或未找到设备: " << GetErrorString(emStatus) << std::endl
    //              << "请检查:1) 相机上电 2) USB3.0 口 3) 设备权限" << std::endl;

        spdlog::error("枚举失败或未找到设备: {}", GetErrorString(emStatus));
        spdlog::error("请检查:1) 相机上电 2) USB3.0 口 3) 设备权限");
        return {};
    }

    std::vector<DeviceInfo> devices;
    spdlog::info("共找到 {} 台设备:", device_num);
    for (uint32_t i = 1; i <= device_num; ++i) { // 序号从 1 开始!
        GX_DEVICE_INFO info;
        //从 info 指向的地址开始,连续 sizeof(GX_DEVICE_INFO) 个字节,每个字节都设置成 0
        memset(&info, 0, sizeof(GX_DEVICE_INFO));
        emStatus = GXGetDeviceInfo(i, &info);
        if (emStatus != GX_STATUS_SUCCESS) {
            spdlog::error("获取设备信息失败: {}", GetErrorString(emStatus));
            continue;
        }
        // 本项目只支持 USB3 相机(U3V);GigE 相机序列号在 stGigEDevInfo 里
        if (info.emDevType == GX_DEVICE_CLASS_U3V) {
            auto &u3v = info.DevInfo.stU3VDevInfo;
            // SDK 里这些字段是 unsigned char[64],转成 std::string 需要显式强转
            spdlog::info("[{}] 型号: {}  序列号: {}", i,
                         reinterpret_cast<const char *>(u3v.chModelName),
                         reinterpret_cast<const char *>(u3v.chSerialNumber));



            devices.push_back({reinterpret_cast<const char *>(u3v.chSerialNumber),
                               reinterpret_cast<const char *>(u3v.chModelName)});
        }
    }
    return devices;
}

// ------------------------------------------------------------------
// 按序列号找设备序号,返回大恒 SDK 的 1-based 设备号;找不到返回 -1
// ------------------------------------------------------------------
int FindDeviceIndexBySerial(const std::vector<DeviceInfo> &devices, const std::string &serial) {
    for (size_t i = 0; i < devices.size(); ++i) {
        if (devices[i].serial == serial) {
            return static_cast<int>(i) + 1; // 转回 1 开始的大恒设备序号
        }
    }
    return -1;
}

// ------------------------------------------------------------------
// 设置参数:在当前值基础上加一个增量,并夹在合法区间内
// ------------------------------------------------------------------
bool AddExposureTime(GX_DEV_HANDLE device, double delta_us) {
    GX_FLOAT_VALUE node;
    memset(&node, 0, sizeof(GX_FLOAT_VALUE));
    GX_STATUS emStatus = GXGetFloatValue(device, "ExposureTime", &node);
    if (emStatus != GX_STATUS_SUCCESS) { spdlog::error("Get ExposureTime: {}", GetErrorString(emStatus)); return false; }

    double value = node.dCurValue + delta_us;
    value = value < 1.0 ? 1.0 : (value > 10000.0 ? 10000.0 : value);

    emStatus = GXSetFloatValue(device, "ExposureTime", value);
    if (emStatus != GX_STATUS_SUCCESS) { spdlog::error("Set ExposureTime: {}", GetErrorString(emStatus)); return false; }
    spdlog::info("曝光时间 -> {} us", value);
    return true;
}

bool AddGain(GX_DEV_HANDLE device, double delta_db) {
    GX_FLOAT_VALUE node;
    memset(&node, 0, sizeof(GX_FLOAT_VALUE));
    GX_STATUS emStatus = GXGetFloatValue(device, "Gain", &node);
    if (emStatus != GX_STATUS_SUCCESS) { spdlog::error("Get Gain: {}", GetErrorString(emStatus)); return false; }

    double value = node.dCurValue + delta_db;
    value = value < 0.0 ? 0.0 : (value > 32.0 ? 32.0 : value);

    emStatus = GXSetFloatValue(device, "Gain", value);
    if (emStatus != GX_STATUS_SUCCESS) { spdlog::error("Set Gain: {}", GetErrorString(emStatus)); return false; }
    spdlog::info("增益 -> {} dB", value);
    return true;
}

bool AddGamma(GX_DEV_HANDLE device, double delta) {
    GX_FLOAT_VALUE node;
    memset(&node, 0, sizeof(GX_FLOAT_VALUE));
    GX_STATUS emStatus = GXGetFloatValue(device, "Gamma", &node);
    if (emStatus != GX_STATUS_SUCCESS) { spdlog::error("Get Gamma: {}", GetErrorString(emStatus)); return false; }

    double value = node.dCurValue + delta;
    value = value < 0.1 ? 0.1 : (value > 3.0 ? 3.0 : value);

    emStatus = GXSetFloatValue(device, "Gamma", value);
    if (emStatus != GX_STATUS_SUCCESS) { spdlog::error("Set Gamma: {}", GetErrorString(emStatus)); return false; }
    spdlog::info("伽马 -> {}", value);
    return true;
}

// ------------------------------------------------------------------
// main
// ------------------------------------------------------------------
int main(int argc, char *argv[]) {


    spdlog::set_pattern("[%H:%M:%S.%e] [%^%l%$] %v");
    spdlog::set_level(spdlog::level::debug);


    // 1. 初始化库
    //    GX_STATUS 是 SDK 的返回码类型;
    //    GXInitLib 返回成功后才允许调用其他 Galaxy 函数;
    //    GXCloseLib 和初始化成对出现。
    GX_STATUS emStatus = GXInitLib();
    spdlog::info("GXInitLib status = {}", emStatus);

    // 2. 枚举设备
    auto devices = EnumerateDevices();
    if (devices.empty()) {
        GXCloseLib();
        return 1;
    }

    // 3. 选设备:命令行给了序列号就按序列号找,没给就用第 1 台
    int device_index = 1;
    std::string serial = devices.front().serial;
    if (argc < 2) {
        spdlog::info("未指定序列号,自动使用第 1 台相机: {}", serial);
    } else {
        serial = argv[1];
        device_index = FindDeviceIndexBySerial(devices, serial);
        if (device_index < 0) {
            spdlog::error("serial not found: {}", serial);
            GXCloseLib();
            return 1;
        }
    }

    // 4. 打开设备
    GX_DEV_HANDLE device = nullptr;
    emStatus = GXOpenDeviceByIndex(device_index, &device);
    if (emStatus != GX_STATUS_SUCCESS) {
        spdlog::error("GXOpenDeviceByIndex: {}", GetErrorString(emStatus));
        GXCloseLib();
        return -1;
    }
    spdlog::info("打开相机 {} 成功", serial);

    // 配置相机(对照 daheng_camera_stream.cpp 构造函数 + ConnectCamera):
    // ------------------------------------------------------------------
    // 1. 自动挡全关
    GXSetEnumValueByString(device, "ExposureAuto", "Off");
    GXSetEnumValueByString(device, "GainAuto", "Off");
    GXSetEnumValueByString(device, "BalanceWhiteAuto", "Continuous");

    // 2. 连续采集 + 关触发
    emStatus = GXSetEnumValueByString(device, "AcquisitionMode", "Continuous");
    if (emStatus != GX_STATUS_SUCCESS) spdlog::error("AcquisitionMode: {}", GetErrorString(emStatus));
    emStatus = GXSetEnumValueByString(device, "TriggerMode", "Off");
    if (emStatus != GX_STATUS_SUCCESS) spdlog::error("TriggerMode: {}", GetErrorString(emStatus));

    // 3. 像素格式:传感器原始数据 Bayer RG8
    emStatus = GXSetEnumValue(device, "PixelFormat", GX_PIXEL_FORMAT_BAYER_RG8);
    if (emStatus != GX_STATUS_SUCCESS) spdlog::error("PixelFormat: {}", GetErrorString(emStatus));

    // 4. 把 DeviceLinkThroughputLimit 拉到最大值,
    GX_INT_VALUE limit_node;
    memset(&limit_node, 0, sizeof(GX_INT_VALUE));
    if (GXGetIntValue(device, "DeviceLinkThroughputLimit", &limit_node) == GX_STATUS_SUCCESS) {
        emStatus = GXSetIntValue(device, "DeviceLinkThroughputLimit", limit_node.nMax);
        if (emStatus != GX_STATUS_SUCCESS)
            spdlog::error("DeviceLinkThroughputLimit: {}", GetErrorString(emStatus));
    }

    // 5. 读分辨率(设置 PixelFormat 之后读,分辨率会随格式变化)
    GX_INT_VALUE width_node, height_node;
    memset(&width_node, 0, sizeof(GX_INT_VALUE));
    memset(&height_node, 0, sizeof(GX_INT_VALUE));
    GXGetIntValue(device, "Width", &width_node);
    GXGetIntValue(device, "Height", &height_node);
    int width = static_cast<int>(width_node.nCurValue);
    int height = static_cast<int>(height_node.nCurValue);
    spdlog::info("分辨率 {}x{}", width, height);

    // 6. 大恒的"流"概念:payload(单帧字节数)要从数据流句柄查询。
    //    项目只取 1 号流,多相机场景需要对应到正确的流下标。
    uint32_t stream_num = 0;
    GX_DS_HANDLE stream_handle = nullptr;
    if (GXGetDataStreamNumFromDev(device, &stream_num) != GX_STATUS_SUCCESS || stream_num < 1) {
        spdlog::error("获取数据流失败");
        GXCloseDevice(device);
        GXCloseLib();
        return -1;
    }
    GXGetDataStreamHandleFromDev(device, 1, &stream_handle);
    uint32_t payload_size = 0;
    GXGetPayLoadSize(stream_handle, &payload_size);

    // 7. 设置 SDK 内部取流缓存个数(5 个),并分配转换缓冲区
    GXSetAcqusitionBufferNumber(device, 5);
    const unsigned int buffer_size = sizeof(unsigned char) * width * height * 3;
    unsigned char *rgb_buffer = static_cast<unsigned char *>(malloc(buffer_size));
    if (rgb_buffer == nullptr) {
        spdlog::error("分配 rgb 缓冲区失败");
        GXCloseDevice(device);
        GXCloseLib();
        return -1;
    }

    // 8. 做 Bayer 转换前要先查相机的滤镜排列(PixelColorFilter 节点):
    //    RG / GB / GR / BG 四种排列,转换函数必须传对,否则颜色错乱。
    int64_t color_filter = GX_COLOR_FILTER_NONE;
    GX_ENUM_VALUE filter_value;
    memset(&filter_value, 0, sizeof(GX_ENUM_VALUE));
    if (GXGetEnumValue(device, "PixelColorFilter", &filter_value) == GX_STATUS_SUCCESS) {
        color_filter = filter_value.stCurValue.nCurValue;
    }

    // 9. 开始取流
    emStatus = GXStreamOn(device);
    if (emStatus != GX_STATUS_SUCCESS) {
        spdlog::error("GXStreamOn: {}", GetErrorString(emStatus));
        free(rgb_buffer);
        GXCloseDevice(device);
        GXCloseLib();
        return -1;
    }

    // 帧率统计
    int frame_count = 0;
    auto fps_window_start = std::chrono::steady_clock::now();
    double fps = 0.0;

    // 循环取帧
    PGX_FRAME_BUFFER frame_buffer = nullptr;
    spdlog::info("取流开始,按 e/d 调曝光、a/q 调增益、s/w 调伽马、ESC 退出");

    while (true) {
        // 1000ms 超时取一帧(DQ = DeQueue,从 SDK 队列取出一帧)
        emStatus = GXDQBuf(device, &frame_buffer, 1000);
        if (emStatus != GX_STATUS_SUCCESS) {
            spdlog::error("GXDQBuf 超时/失败: {}", GetErrorString(emStatus));
            break; // 练习程序直接退出;主工程在这里做断流重连(见 1.3.4)
        }

        // 帧状态检查:丢包、传输错误的帧要跳过,但必须照常 QB 还回去!
        if (frame_buffer->nStatus != GX_FRAME_STATUS_SUCCESS) {
            spdlog::error("帧状态异常: {:#x}", static_cast<uint32_t>(frame_buffer->nStatus));
            GXQBuf(device, frame_buffer); // 无论帧好坏都要还,否则队列会被掏空
            continue;
        }

        // Bayer RG8 -> BGR24。参数含义:
        //   RAW2RGB_NEIGHBOUR   邻域插值(简单快速;官方还有 2x2/3x3 等可选)
        //   DX_PIXEL_COLOR_FILTER(滤镜排列) 由 PixelColorFilter 节点查询得到
        //   false                不翻转图像
        //   DX_ORDER_BGR         输出 OpenCV 的 BGR 通道序
        VxInt32 dx_status = DxRaw8toRGB24Ex(frame_buffer->pImgBuf, rgb_buffer,
                                            frame_buffer->nWidth, frame_buffer->nHeight,
                                            RAW2RGB_NEIGHBOUR, DX_PIXEL_COLOR_FILTER(color_filter),
                                            false, DX_ORDER_BGR);
        if (dx_status != DX_OK) {
            spdlog::error("DxRaw8toRGB24Ex 失败: {:#x}", static_cast<uint32_t>(dx_status));
            GXQBuf(device, frame_buffer); // 同样要还
            continue;
        }

        // 图像显示
        cv::Mat image(frame_buffer->nHeight, frame_buffer->nWidth, CV_8UC3, rgb_buffer);
        cv::Mat display = image.clone();

        ++frame_count;
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - fps_window_start).count();
        if (elapsed >= 1.0) {
            fps = frame_count / elapsed;
            frame_count = 0;
            fps_window_start = now;
        }
        cv::putText(display, "FPS: " + std::to_string(fps), cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
        cv::imshow("daheng", display);

        bool quit = false;
        switch (cv::waitKey(1)) {
            case 27:  quit = true; break;              // ESC
            case 'e': AddExposureTime(device, 25.0); break;  // 增大曝光
            case 'd': AddExposureTime(device, -25.0); break; // 减小曝光
            case 'a': AddGain(device, 0.1); break;           // 增大增益
            case 'q': AddGain(device, -0.1); break;          // 减小增益
            case 's': AddGamma(device, 0.1); break;          // 增大伽马
            case 'w': AddGamma(device, -0.1); break;         // 减小伽马
            default: break;
        }

        // 用完后把帧还回 SDK 队列(QB = EnQueue),漏还的症状与海康相同:
        // 队列被掏空,GXDQBuf 永远超时,帧率掉到 0
        GXQBuf(device, frame_buffer);
        if (quit) break;
    }

    // 逆序关闭:停流 -> 释放缓冲 -> 关设备 -> 关库(与 GXInitLib 配对)
    GXStreamOff(device);
    free(rgb_buffer);
    GXCloseDevice(device);
    GXCloseLib();
    spdlog::info("相机已关闭");
    return 0;
}