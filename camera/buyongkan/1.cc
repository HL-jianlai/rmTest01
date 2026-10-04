#include <sdk/GxIAPI.h>

#include <iostream>


//初始化库
int main() {

    GX_STATUS status = GXInitLib();
        //GX_STATUS 是 SDK 的返回码类型；
        //GXInitLib 返回成功后才允许调用其他 Galaxy 函数；
        //GXCloseLib 和初始化成对出现。
    std::cout << "GXInitLib status = " << status << std::endl;
    if (status == GX_STATUS_SUCCESS) {
    GXCloseLib();
    }
    
    return status == GX_STATUS_SUCCESS ? 0 : 1;

}

