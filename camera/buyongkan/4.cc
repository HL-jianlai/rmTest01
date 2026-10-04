#include <sdk/GxIAPI.h>
#include <vector>
#include <cstring>

#include <iostream>

struct DeviceInfo {
    std::string serial;
    std::string model;
};

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

std::vector<DeviceInfo> EnumerateDevices() {
    uint32_t device_num = 0;
    // 第二个参数是枚举超时(ms):GigE 相机响应慢,官方建议至少 1000
    GX_STATUS emStatus = GXUpdateAllDeviceList(&device_num, 1000);
    if (emStatus != GX_STATUS_SUCCESS || device_num == 0) {
        std::cout << "枚举失败或未找到设备: " << GetErrorString(emStatus) << std::endl
                  << "请检查:1) 相机上电 2) USB3.0 口 3) 设备权限" << std::endl;
        return {};
    }

    std::vector<DeviceInfo> devices;
    std::cout << "共找到 " << device_num << " 台设备:" << std::endl;
    for (uint32_t i = 1; i <= device_num; ++i) { // 序号从 1 开始!
        GX_DEVICE_INFO info;
        memset(&info, 0, sizeof(GX_DEVICE_INFO));
        emStatus = GXGetDeviceInfo(i, &info);
        if (emStatus != GX_STATUS_SUCCESS) {
            continue;
        }
        
        if (info.emDevType == GX_DEVICE_CLASS_U3V) {
            auto &u3v = info.DevInfo.stU3VDevInfo;
            // SDK 里这些字段是 unsigned char[64],转成 std::string 需要显式强转
            std::cout << "  [" << i << "] 型号: " << reinterpret_cast<const char *>(u3v.chModelName)
                      << "  序列号: " << reinterpret_cast<const char *>(u3v.chSerialNumber) << std::endl;
            devices.push_back({reinterpret_cast<const char *>(u3v.chSerialNumber),
                               reinterpret_cast<const char *>(u3v.chModelName)});
        }
    }
    return devices;
}

int FindDeviceIndexBySerial(const std::vector<DeviceInfo> &devices, const std::string &serial) {
    for (size_t i = 0; i < devices.size(); ++i) {
        if (devices[i].serial == serial) {
            return static_cast<int>(i) + 1; // 转回 1 开始的大恒设备序号
        }
    }
    return -1;
}

int main(int argc, char *argv[]) {

    GX_STATUS status = GXInitLib();
    std::cout << "GXInitLib status = " << status << std::endl;
    
    auto devices = EnumerateDevices();
    if (devices.empty()) {
    GXCloseLib();
    return 1;
    }

    if (argc < 2) {
    std::cout << "用法: ./daheng_demo <相机序列号>" << std::endl;
    GXCloseLib();
    return 0;
    }
     
    const std::string serial = argv[1];

    int device_index = FindDeviceIndexBySerial(devices, serial);
    if (device_index < 0) {
    std::cout << "serial not found" << std::endl;
    GXCloseLib();
    return 1;
    }

    return status == GX_STATUS_SUCCESS ? 0 : 1;

}