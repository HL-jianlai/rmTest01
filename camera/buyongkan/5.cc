

//设置参数
bool AddExposureTime(GX_DEV_HANDLE device, double delta_us) {
    GX_FLOAT_VALUE node;
    memset(&node, 0, sizeof(GX_FLOAT_VALUE));
    GX_STATUS emStatus = GXGetFloatValue(device, "ExposureTime", &node);
    if (emStatus != GX_STATUS_SUCCESS) { std::cout << "Get ExposureTime: " << GetErrorString(emStatus) << std::endl; return false; }

    double value = node.dCurValue + delta_us;
    value = value < 1.0 ? 1.0 : (value > 10000.0 ? 10000.0 : value);

    emStatus = GXSetFloatValue(device, "ExposureTime", value);
    if (emStatus != GX_STATUS_SUCCESS) { std::cout << "Set ExposureTime: " << GetErrorString(emStatus) << std::endl; return false; }
    std::cout << "曝光时间 -> " << value << " us" << std::endl;
    return true;
}