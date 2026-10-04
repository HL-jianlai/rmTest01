#include <sdk/GxIAPI.h>

#include <iostream>

//枚举设备
int main() {

    GX_STATUS status = GXInitLib();
    std::cout << "GXInitLib status = " << status << std::endl;
    
    auto devices = EnumerateDevices();
    if (devices.empty()) {
    GXCloseLib();
    return 1;
    }
    
    if (status == GX_STATUS_SUCCESS) {
    GXCloseLib();
    }
    
    return status == GX_STATUS_SUCCESS ? 0 : 1;

}