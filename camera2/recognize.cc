// 大恒(Galaxy)工业相机 + OpenCV 装甲板(armor)识别
//
// 思路(BGR/HSV 双通道判定灯条颜色):
//   1. 取流 -> Bayer RG8 转 BGR24,拿到 OpenCV 的 cv::Mat
//   2. HSV 通道做颜色阈值,红色灯条要拆成两段色相区间([0,10] ∪ [160,180])
//      蓝色灯条落在 [100,130];S/V 双阈值用来压掉暗部和低饱和背景
//   3. 形态学闭运算把 LED 灯珠连成整根灯条,再开运算去噪点
//      (这一步无条件执行,不受"是否显示二值图"影响)
//   4. findContours + minAreaRect,用 长宽比 / 填充率 / 角度 / 亮度 筛出灯条候选
//      (刻意不限制灯条的像素大小,远近目标一视同仁)
//   5. 同色灯条两两配对:中心距、灯条长度比、朝向夹角、位置关系打分
//   6. 角点从灯条轮廓上取真正的外沿极值点(不是 minAreaRect 的角),
//      即左灯条最左侧上下两点、右灯条最右侧上下两点,画框画点
//   7. 输出识别帧率、算法耗时、颜色(Red/Blue)
//
// 编译:见同目录 CMakeLists.txt
// 用法:./recognize [相机序列号]

#include <sdk/GxIAPI.h>
#include <sdk/DxImageProc.h>

#include <opencv2/opencv.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include <spdlog/spdlog.h>

// ==================================================================
// 一、可调参数(全部集中在这里,现场调试只改这一块)
// ==================================================================

// ---- 颜色阈值:红蓝灯条 HSV ----
//
// 关键前提:灯条点亮后**芯部是过曝的**。传感器在芯部三通道全部打满,
// 颜色被冲淡 —— 蓝色灯条芯部变成 B=G=255 的青色(H=90),
// 红色灯条芯部变成 R=G=255 的橙黄色(H≈11~20)。
// 如果色相窗口只按"纯色"来定(H_red≤10、H_blue≥95),
// 芯部整片被拒,掩膜就变成一圈空心的环。环一旦断开,
// minAreaRect 会缩成细条、填充率暴跌,灯条就再也检不出来,
// 或者碎成一颗颗灯珠。实测:红色芯部 H=11~20,蓝色芯部 H=90、S 只有 129~175。
// 所以窗口必须往"过曝方向"放开,判据靠 V(亮)而不是靠 S(纯)。
namespace ColorCfg {
constexpr int kRedHMin1   = 0;    // 红:色相环绕,拆两段
constexpr int kRedHMax1   = 15;   // 必须覆盖过曝芯部的 H=11~20(实测)
constexpr int kRedHMin2   = 160;
constexpr int kRedHMax2   = 179;
constexpr int kRedSMin    = 120;  // 饱和度下限,压掉灰白背景
constexpr int kRedVMin    = 100;  // 明度下限,压掉暗部

constexpr int kBlueHMin   = 85;   // 必须覆盖过曝芯部的 H=90(实测)
constexpr int kBlueHMax   = 140;
// S 别再往上加:蓝色芯部实测只有 129~175,调到 170 会把整条灯条拒掉。
// 也别往下调太多:实测 S=125 就开始冒出新的杂散连通域,150 是拐点。
constexpr int kBlueSMin   = 150;
constexpr int kBlueVMin   = 100;
}  // namespace ColorCfg

// ---- 灯条几何筛选 ----
// 主判据是"长宽比 + 填充率 + 亮度"这三个相对量,远近大小的灯条都能进得来,
// 不按像素尺寸卡远近。但光有相对量不够,必须再配一个很低的绝对下限 kMinArea
// 把零星噪点挡在外面 —— 原因见 DetectLightBars 里的说明。
namespace BarCfg {
// 绝对尺寸下限,很低,只为滤掉零星噪点。必须存在:其余关卡全是比值、与尺度无关,
// 没有它的话 3x2 的噪点也能配出高分(见 DetectLightBars 里的详细说明)。
constexpr double kMinArea   = 60.0;   // 轮廓面积(像素)
constexpr float  kMinLength = 12.0f;  // 灯条长边(像素)
constexpr float kMinRatio  = 1.5f;    // 长/宽 比下限(灯条是细长条)
// 长/宽 比上限。这个值要给 yaw 留足余量:装甲板绕 yaw 转 φ 后,灯条的横向厚度
// 按 cosφ 收缩,而长度基本不变,长宽比就按 1/cosφ 涨上去 —— 正对时 7.1,
// 40° 时 11.0,60° 已经 18.0,65° 20.5,70° 27.4,80° 50.6。
// 原来写 18 等于"转 60° 以上必毙",而且灯条的填充率(0.91~0.95)和亮度(229)
// 此时全都健康 —— 被毙掉的是一根几何上完全正常的灯条,纯粹吃亏在薄。
// 放宽到 40 可覆盖到约 80°;再往上灯条只剩几像素宽,掩膜本身就撑不住了。
constexpr float kMaxRatio  = 40.0f;   // 长/宽 比上限
constexpr float kMinBright = 130.0;// 灯条自身像素的平均 V,LED 一定比背景亮
constexpr float kMinFill   = 0.25f;   // contour 面积 / minAreaRect 面积
// 实心灯条的填充率天然接近 1.0,这里不设上限,只保留填充率下限。
// 不要往下调这个值:抗锯齿边缘 + 噪声会让 mask 比 minAreaRect 略小,
// 实测真实灯条填充率就在 0.89~0.90 之间,写 0.90 会正好卡在边界上,
// 时检出时漏检,而且长宽比越接近直角越容易踩到。
constexpr float kMaxFill   = 1.00f;
}  // namespace BarCfg

// ---- 灯条配对(装甲板) ----
namespace PairCfg {
constexpr float kMaxAngleDiff     = 25.0f;  // 两灯条长轴夹角上限(度)
constexpr float kMaxLengthDiffR   = 0.8f;   // |L1-L2| / ((L1+L2)/2) 上限
constexpr float kMaxYDiffRatio    = 0.5f;// |dy| / 灯条平均长度
constexpr float kMinCenterDist    = 1.0f;// 两灯条中心距下限
// 中心距 / 平均长度,约等于装甲板高宽比。
// 下限要留足余量:装甲板绕 yaw 转起来后,两灯条中心的水平间距按 cos(yaw) 收缩,
// 而灯条长度基本不变,这个比值会一路变小。0.4 只能撑到约 65°,
// 0.25 可以覆盖到约 80°,再大就是几乎侧对镜头了,没有识别价值。
constexpr float kMinDistRatio     = 0.25f;
constexpr float kMaxDistRatio     = 5.5f;
constexpr float kSmallArmorRatio  = 3.2f;   // 超过它按"小装甲"给高分
constexpr float kScoreThreshold   = 0.30f;  // 配对得分低于此值丢弃
constexpr float kWeightAngle      = 0.30f;
constexpr float kWeightLength     = 0.25f;
constexpr float kWeightDist       = 0.30f;
constexpr float kWeightY          = 0.15f;
}  // namespace PairCfg

constexpr int kRed = 0;
constexpr int kBlue = 1;

inline const char *ColorName(int color) { return color == kRed ? "Red" : "Blue"; }
inline cv::Scalar ColorDraw(int color) { return color == kRed ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 128, 0); }

// ==================================================================
// 二、数据结构
// ==================================================================

struct DeviceInfo {
    std::string serial;
    std::string model;
};

struct LightBar {
    cv::RotatedRect rect;
    std::vector<cv::Point2f> points;  // 4 个角点
    cv::Point2f center;
    cv::Point2f left_top, left_bottom;    // 灯条自身靠左的两个角点
    cv::Point2f right_top, right_bottom;  // 灯条自身靠右的两个角点
    float length = 0.0f;  // 长边
    float width = 0.0f;   // 短边
    float angle = 0.0f;   // 长轴方向,归一化到 (-90, 90]
    int color = kBlue;
    float brightness = 0.0f;
    float score = 0.0f;
};

struct ArmorPlate {
    int color = kBlue;
    float score = 0.0f;
    cv::Point2f center;                // 四个角点的形心(梯形的视觉中心)
    std::vector<cv::Point2f> corners;  // 顺序:左上、右上、右下、左下
    float left_edge = 0.0f;            // 左灯条外沿边长(梯形左边)
    float right_edge = 0.0f;           // 右灯条外沿边长(梯形右边)
    float trape_ratio = 1.0f;          // 短边/长边 ∈ (0,1],1.0 = 正对镜头
    LightBar left;
    LightBar right;
    int id = -1;  // 跨帧稳定编号,便于观察跟踪
};

// ==================================================================
// 三、大恒相机部分(与 camera/daheng.cc 保持一致)
// ==================================================================

std::string GetErrorString(GX_STATUS emErrorStatus) {
    char *error_info = nullptr;
    size_t size = 0;
    GX_STATUS emStatus = GXGetLastError(&emErrorStatus, nullptr, &size);
    if (emStatus != GX_STATUS_SUCCESS) {
        return "<Error when calling GXGetLastError>";
    }
    error_info = new char[size];
    emStatus = GXGetLastError(&emErrorStatus, error_info, &size);
    std::string error_string = error_info != nullptr ? error_info : "";
    delete[] error_info;
    return emStatus == GX_STATUS_SUCCESS ? error_string : "<Error when calling GXGetLastError>";
}

std::vector<DeviceInfo> EnumerateDevices() {
    uint32_t device_num = 0;
    GX_STATUS emStatus = GXUpdateAllDeviceList(&device_num, 1000);
    if (emStatus != GX_STATUS_SUCCESS || device_num == 0) {
        spdlog::error("枚举失败或未找到设备: {}", GetErrorString(emStatus));
        spdlog::error("请检查:1) 相机上电 2) USB3.0 口 3) 设备权限");
        return {};
    }

    std::vector<DeviceInfo> devices;
    spdlog::info("共找到 {} 台设备:", device_num);
    for (uint32_t i = 1; i <= device_num; ++i) {
        GX_DEVICE_INFO info;
        memset(&info, 0, sizeof(GX_DEVICE_INFO));
        emStatus = GXGetDeviceInfo(i, &info);
        if (emStatus != GX_STATUS_SUCCESS) {
            spdlog::error("获取设备信息失败: {}", GetErrorString(emStatus));
            continue;
        }
        if (info.emDevType == GX_DEVICE_CLASS_U3V) {
            auto &u3v = info.DevInfo.stU3VDevInfo;
            spdlog::info("[{}] 型号: {}  序列号: {}", i,
                         reinterpret_cast<const char *>(u3v.chModelName),
                         reinterpret_cast<const char *>(u3v.chSerialNumber));
            devices.push_back({reinterpret_cast<const char *>(u3v.chSerialNumber),
                               reinterpret_cast<const char *>(u3v.chModelName)});
        }
    }
    return devices;
}

int FindDeviceIndexBySerial(const std::vector<DeviceInfo> &devices, const std::string &serial) {
    for (size_t i = 0; i < devices.size(); ++i) {
        if (devices[i].serial == serial) {
            return static_cast<int>(i) + 1;
        }
    }
    return -1;
}

bool AddExposureTime(GX_DEV_HANDLE device, double delta_us) {
    GX_FLOAT_VALUE node;
    memset(&node, 0, sizeof(GX_FLOAT_VALUE));
    GX_STATUS emStatus = GXGetFloatValue(device, "ExposureTime", &node);
    if (emStatus != GX_STATUS_SUCCESS) {
        spdlog::error("Get ExposureTime: {}", GetErrorString(emStatus));
        return false;
    }
    // 夹到相机自己报的上下限,不再硬编码 1~10000us:
    // 写死的话相机支持更长曝光时按 e 到 10000 就上不去了
    double value = node.dCurValue + delta_us;
    value = value < node.dMin ? node.dMin : (value > node.dMax ? node.dMax : value);
    emStatus = GXSetFloatValue(device, "ExposureTime", value);
    if (emStatus != GX_STATUS_SUCCESS) {
        spdlog::error("Set ExposureTime: {}", GetErrorString(emStatus));
        return false;
    }
    // 回读:相机会把请求值吸附到步长 dInc、并夹住范围,
    // 直接打印 value 是"我想设的值",不是真正生效的值
    GX_FLOAT_VALUE after;
    memset(&after, 0, sizeof(GX_FLOAT_VALUE));
    if (GXGetFloatValue(device, "ExposureTime", &after) != GX_STATUS_SUCCESS) {
        spdlog::info("曝光时间 -> {:.0f} us(回读失败,这是请求值)", value);
        return true;
    }
    const char *lim = after.dCurValue <= node.dMin + 1e-9    ? "  [已到下限]"
                      : after.dCurValue >= node.dMax - 1e-9  ? "  [已到上限]"
                                                             : "";
    spdlog::info("曝光时间 -> {:.0f} us  (范围 {:.0f}~{:.0f}, 步长 {:.0f}){}", after.dCurValue,
                 node.dMin, node.dMax, node.dInc, lim);
    return true;
}

bool AddGain(GX_DEV_HANDLE device, double delta_db) {
    GX_FLOAT_VALUE node;
    memset(&node, 0, sizeof(GX_FLOAT_VALUE));
    GX_STATUS emStatus = GXGetFloatValue(device, "Gain", &node);
    if (emStatus != GX_STATUS_SUCCESS) {
        spdlog::error("Get Gain: {}", GetErrorString(emStatus));
        return false;
    }
    double value = node.dCurValue + delta_db;
    value = value < node.dMin ? node.dMin : (value > node.dMax ? node.dMax : value);
    emStatus = GXSetFloatValue(device, "Gain", value);
    if (emStatus != GX_STATUS_SUCCESS) {
        spdlog::error("Set Gain: {}", GetErrorString(emStatus));
        return false;
    }
    // 同曝光:回读真实生效值。0.1dB 是浮点累加(0.1+0.1+0.1=0.30000000000000004),
    // 打印累加结果既不准也难看
    GX_FLOAT_VALUE after;
    memset(&after, 0, sizeof(GX_FLOAT_VALUE));
    if (GXGetFloatValue(device, "Gain", &after) != GX_STATUS_SUCCESS) {
        spdlog::info("增益 -> {:.2f} dB(回读失败,这是请求值)", value);
        return true;
    }
    const char *lim = after.dCurValue <= node.dMin + 1e-9    ? "  [已到下限]"
                      : after.dCurValue >= node.dMax - 1e-9  ? "  [已到上限]"
                                                             : "";
    spdlog::info("增益 -> {:.2f} dB  (范围 {:.2f}~{:.2f}, 步长 {:.2f}){}", after.dCurValue,
                 node.dMin, node.dMax, node.dInc, lim);
    return true;
}

// ==================================================================
// 四、灯条检测
// ==================================================================

// 生成某一颜色的二值图。红色要合并两段色相区间(0 附近环绕)。
// 形态学无条件执行 —— 这里踩过坑:早先把它挂在"是否显示二值图"的开关上,
// 结果是关掉显示时(程序默认状态)灯条不做闭运算,LED 灯珠连不成整根,
// 每颗灯珠单独去做长宽比筛选全部被毙掉,一根灯条都检不出。
// 检测结果绝不能取决于用户有没有按 b 键。
cv::Mat BuildColorMask(const cv::Mat &hsv, int color) {
    cv::Mat mask;
    if (color == kRed) {
        cv::Mat m1, m2;
        cv::inRange(hsv, cv::Scalar(ColorCfg::kRedHMin1, ColorCfg::kRedSMin, ColorCfg::kRedVMin),
                    cv::Scalar(ColorCfg::kRedHMax1, 255, 255), m1);
        cv::inRange(hsv, cv::Scalar(ColorCfg::kRedHMin2, ColorCfg::kRedSMin, ColorCfg::kRedVMin),
                    cv::Scalar(ColorCfg::kRedHMax2, 255, 255), m2);
        cv::bitwise_or(m1, m2, mask);
    } else {
        cv::inRange(hsv, cv::Scalar(ColorCfg::kBlueHMin, ColorCfg::kBlueSMin, ColorCfg::kBlueVMin),
                    cv::Scalar(ColorCfg::kBlueHMax, 255, 255), mask);
    }

    // 竖直方向长核:把断开的 LED 灯珠连成一根完整灯条
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE,
                     cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 9)));
    // 这里原本还有一次 MORPH_OPEN(3x3),已经去掉 —— 它会啃断细灯条:
    // 灯条只有几像素宽时,腰上不到 3px 的地方被 3x3 的腐蚀整个吃掉,
    // 一根灯条断成两截。实测蓝色装甲板缩到 0.18(灯条 7px 宽)时,
    // 闭运算已经把它接成完整一根,紧接着的开运算又在细腰处切回两段,
    // 于是同一根灯条的上半截和下半截各自去配对,凭空多出一块装甲板。
    // 去噪点的职责现在由 kMinArea(轮廓面积下限)承担,开运算是冗余的。
    return mask;
}

// 把长轴角度归一化到 (-90, 90],方便比较两根灯条是否平行
float NormalizeAngle(float deg) {
    while (deg > 90.0f) deg -= 180.0f;
    while (deg <= -90.0f) deg += 180.0f;
    return deg;
}

// 现场量测:把画面里"亮且饱和"的像素按色相分箱打出来。
// 调颜色阈值不该靠猜 —— 把灯条摆进画面按 p,看它的 H/S/V 落在哪个区间,
// 再照着改 ColorCfg。蓝色灯条最容易在这里翻车:色相偏青一点就会漏。
void DumpBrightHue(const cv::Mat &hsv) {
    int hist[180] = {0};
    long bright = 0, saturated = 0;
    int max_s = 0, min_s = 255, max_v = 0;
    for (int y = 0; y < hsv.rows; ++y) {
        const cv::Vec3b *row = hsv.ptr<cv::Vec3b>(y);
        for (int x = 0; x < hsv.cols; ++x) {
            const int H = row[x][0], S = row[x][1], V = row[x][2];
            if (V < BarCfg::kMinBright) continue;   // 只看够亮的
            ++bright;
            if (S < 90) continue;                   // 低饱和的是白/灰,不是 LED
            ++saturated;
            ++hist[H];
            max_s = std::max(max_s, S);
            min_s = std::min(min_s, S);
            max_v = std::max(max_v, V);
        }
    }
    spdlog::info("亮像素(V>={}): {} 个,其中 S>=90 的: {} 个", BarCfg::kMinBright, bright, saturated);
    if (saturated == 0) {
        spdlog::info("没有饱和的亮像素 —— 灯条不在画面里,或曝光太低");
        return;
    }
    spdlog::info("这些像素的 S 范围 [{}, {}],V 最大 {}", min_s, max_s, max_v);
    for (int b = 0; b < 180; b += 5) {
        long c = 0;
        for (int k = b; k < b + 5; ++k) c += hist[k];
        if (c == 0) continue;
        std::string bar(static_cast<size_t>(std::min<long>(50, c / 100)), '#');
        spdlog::info("  H={:3d}-{:3d} : {:8d} {}", b, b + 4, c, bar);
    }
    spdlog::info("当前蓝色窗口 H∈[{},{}] S>={} V>={};红色 H∈[{},{}]∪[{},{}] S>={} V>={}",
                 ColorCfg::kBlueHMin, ColorCfg::kBlueHMax, ColorCfg::kBlueSMin, ColorCfg::kBlueVMin,
                 ColorCfg::kRedHMin1, ColorCfg::kRedHMax1, ColorCfg::kRedHMin2, ColorCfg::kRedHMax2,
                 ColorCfg::kRedSMin, ColorCfg::kRedVMin);
}

// 兜底:退化情形下直接把矩形四角按 x 分左右、按 y 分上下
void SplitRectCorners(LightBar &bar) {
    std::vector<cv::Point2f> pts = bar.points;
    std::sort(pts.begin(), pts.end(), [](const cv::Point2f &a, const cv::Point2f &b) {
        return a.x != b.x ? a.x < b.x : a.y < b.y;
    });
    std::vector<cv::Point2f> left_pts = {pts[0], pts[1]};
    std::vector<cv::Point2f> right_pts = {pts[2], pts[3]};
    std::sort(left_pts.begin(), left_pts.end(),
              [](const cv::Point2f &a, const cv::Point2f &b) { return a.y < b.y; });
    std::sort(right_pts.begin(), right_pts.end(),
              [](const cv::Point2f &a, const cv::Point2f &b) { return a.y < b.y; });
    bar.left_top = left_pts[0];
    bar.left_bottom = left_pts[1];
    bar.right_top = right_pts[0];
    bar.right_bottom = right_pts[1];
}

// V 通道上的双线性采样(亚像素)。越界返回 -1,由调用者判断而不是取 0 —— 取 0
// 会在画面边缘凭空造出一条"从亮到黑"的假边,把角点吸到图像边界上。
float SampleValue(const cv::Mat &value_channel, cv::Point2f p) {
    if (p.x < 0.0f || p.y < 0.0f || p.x > value_channel.cols - 1.001f ||
        p.y > value_channel.rows - 1.001f) {
        return -1.0f;
    }
    const int x0 = static_cast<int>(p.x), y0 = static_cast<int>(p.y);
    const float fx = p.x - x0, fy = p.y - y0;
    const uchar *r0 = value_channel.ptr<uchar>(y0);
    const uchar *r1 = value_channel.ptr<uchar>(y0 + 1);
    const float a = r0[x0] * (1.0f - fx) + r0[x0 + 1] * fx;
    const float b = r1[x0] * (1.0f - fx) + r1[x0 + 1] * fx;
    return a * (1.0f - fy) + b * fy;
}

// 从 origin 沿 dir 往外走,找灯条的边缘,返回它到 origin 的亚像素距离。
//
// 判据用【亮度梯度最大处】,不是"亮度降到峰值的某个百分比"。原因实测:
// 灯条核部过曝在 255 削顶,电平判据会被整体亮度拉走(±30% 亮度下漂 1.5px);
// 而梯度的极值位置对任意单调缩放不变(漂 1.0px)。作为对照,现在用的
// "颜色掩膜轮廓边界"在同样扰动下漂 4.8px,而且中间还有 3.6px 的不连续跳变
// —— 那个跳变就是"蓝色角点标注跳一下"的来源。
bool FindBarEdge(const cv::Mat &value_channel, cv::Point2f origin, cv::Point2f dir, float max_s,
                 float *out_s) {
    const float step = 0.5f;
    const int n = static_cast<int>(max_s / step);
    if (n < 4) return false;

    std::vector<float> g(n + 1, 0.0f);
    float gmax = 0.0f;
    for (int i = 1; i <= n; ++i) {
        const float s = i * step;
        const float inner = SampleValue(value_channel, origin + dir * (s - step));
        const float outer = SampleValue(value_channel, origin + dir * (s + step));
        if (inner < 0.0f || outer < 0.0f) break;   // 走到画面外了,不再往外找
        g[i] = std::abs(outer - inner);
        gmax = std::max(gmax, g[i]);
    }
    if (gmax < 12.0f) return false;   // 整条扫描线上没有像样的边

    // 取【第一个】够强的局部极大:往外最近的边才是灯条的边。再往外可能是
    // 装甲板上的数字、别的灯条或高光,不能让它把角点拉走。
    const float thresh = std::max(12.0f, 0.5f * gmax);
    int bi = -1;
    for (int i = 2; i + 1 <= n; ++i) {
        if (g[i] >= thresh && g[i] >= g[i - 1] && g[i] >= g[i + 1]) {
            bi = i;
            break;
        }
    }
    if (bi < 0) {   // 没有明显局部极大(边很平缓),退回全局最大
        bi = 1;
        for (int i = 2; i <= n; ++i) {
            if (g[i] > g[bi]) bi = i;
        }
    }

    // 三点抛物线取亚像素极值
    float delta = 0.0f;
    if (bi >= 2 && bi + 1 <= n) {
        const float g0 = g[bi - 1], g1 = g[bi], g2 = g[bi + 1];
        const float den = g0 - 2.0f * g1 + g2;
        if (std::abs(den) > 1e-4f) delta = 0.5f * (g0 - g2) / den;
        delta = std::max(-1.0f, std::min(1.0f, delta));
    }
    *out_s = bi * step + delta * step;
    return true;
}

// 用亮度梯度定灯条的四个外沿角点。成功返回 true。
//
// 为什么不用颜色掩膜的轮廓:蓝色掩膜的外沿是 S 阈值切出来的等值线,而这条
// 等值线落在灯条自身的 S 渐变【里面】—— 实测它砍掉蓝色灯条自身 24%~36% 的
// 像素(红只有 0.0%~0.1%,所以红一直很稳)。等值线一旦被白平衡/曝光推动,
// 轮廓就从外沿开始被不均匀地啃掉,角点跟着走。亮度的梯度极值则是灯条的
// 物理边缘,对整体缩放不变。
//
// 做法:轮廓只用来定"灯条在哪、朝哪"(长轴方向、中心)。然后在 V 通道上沿
// 垂直方向做若干条扫描线,每条求出左右边缘点(亚像素),左右各最小二乘拟合
// 一条直线,再外推到灯条两端得到四角。灯条的边在 3D 里是直线、透视投影后
// 仍是直线,所以拟合是对的;而单点抖动会被整条边平均掉 —— 角点因此才稳。
bool ExtractCornersByGradient(const cv::Mat &value_channel, LightBar &bar) {
    // 长轴方向,取法与 BarDirection 一致(boxPoints 顺序与直觉相反,见那里的说明)
    cv::Point2f along = (bar.rect.size.width >= bar.rect.size.height)
                            ? bar.points[2] - bar.points[1]
                            : bar.points[1] - bar.points[0];
    const float alen = std::hypot(along.x, along.y);
    if (alen < 1e-3f) return false;
    const cv::Point2f u = along / alen;

    // 横跨灯条的方向:统一取 x 分量为正的一侧当"右",左 = 反向。
    // 这份约定必须和 ExtractOuterCornersByContour 一致 —— FindArmors 组装
    // corners 时依赖 left_*/right_* 的左右关系。
    cv::Point2f dir(-u.y, u.x);
    if (dir.x < 0.0f) dir = -dir;

    const float L = bar.length;
    if (L < 8.0f) return false;

    // 扫描线只覆盖中段 70%:两端有圆角、还可能被画面裁掉,梯度不可靠。
    // 四角靠拟合线外推到 ±L/2,所以不需要扫到端头。
    const int n_scan = std::max(5, std::min(15, static_cast<int>(L / 8.0f)));
    const float t0 = -0.35f * L, t1 = 0.35f * L;
    // 往外走多远:掩膜常比真实灯条胖(闭运算),给点余量
    const float max_s = 1.2f * bar.width + 8.0f;

    std::vector<cv::Point2f> pl, pr;
    pl.reserve(n_scan);
    pr.reserve(n_scan);
    for (int i = 0; i < n_scan; ++i) {
        const float t = t0 + (t1 - t0) * i / (n_scan - 1);
        const cv::Point2f o = bar.center + u * t;
        float s = 0.0f;
        if (FindBarEdge(value_channel, o, -dir, max_s, &s)) pl.push_back(o - dir * s);
        if (FindBarEdge(value_channel, o, dir, max_s, &s)) pr.push_back(o + dir * s);
    }
    if (pl.size() < 4 || pr.size() < 4) return false;   // 边点不够,交给兜底

    // 最小二乘拟合每条边。点写成 center + u*t + dir*ss,拟合 ss = a*t + b。
    auto fit = [&](const std::vector<cv::Point2f> &pts, float *a, float *b) {
        double st = 0, ss = 0, stt = 0, sts = 0;
        const double n = static_cast<double>(pts.size());
        for (const cv::Point2f &p : pts) {
            const cv::Point2f d = p - bar.center;
            const double t = d.dot(u), s = d.dot(dir);
            st += t;
            ss += s;
            stt += t * t;
            sts += t * s;
        }
        const double den = n * stt - st * st;
        if (std::abs(den) < 1e-6) return false;
        *a = static_cast<float>((n * sts - st * ss) / den);
        *b = static_cast<float>((ss - *a * st) / n);
        return true;
    };
    float al = 0, bl = 0, ar = 0, br = 0;
    if (!fit(pl, &al, &bl) || !fit(pr, &ar, &br)) return false;

    // 外推到灯条两端
    const float half = 0.5f * L;
    auto at = [&](float a, float b, float t) {
        return bar.center + u * t + dir * (a * t + b);
    };
    const cv::Point2f l0 = at(al, bl, -half), l1 = at(al, bl, half);
    const cv::Point2f r0 = at(ar, br, -half), r1 = at(ar, br, half);

    // 命名按图像上下(y 小的叫 top),与兜底实现一致
    bar.left_top = l0.y <= l1.y ? l0 : l1;
    bar.left_bottom = l0.y <= l1.y ? l1 : l0;
    bar.right_top = r0.y <= r1.y ? r0 : r1;
    bar.right_bottom = r0.y <= r1.y ? r1 : r0;
    return true;
}

// 兜底:从颜色掩膜的轮廓上取外沿角点。左端最上/最下 + 右端最上/最下。
//
// 什么时候会走到这里:亮度边缘找不齐(梯度太弱、扫描线出画面、灯条太短)。
// 正常帧走的是 ExtractCornersByGradient —— 那条路对曝光/白平衡更稳。
// 保留这一条是为了不出现"新方法失败就整根灯条没角点"的断崖。
//
// 它的固有弱点(也是当初要换掉它的原因):轮廓是阈值等值线,灯条的边一旦
// 被阈值从中啃过,取到的极值点就跟着阈值走。
void ExtractOuterCornersByContour(const std::vector<cv::Point> &contour, LightBar &bar) {
    if (contour.size() < 2) {
        SplitRectCorners(bar);
        return;
    }

    // 长轴方向。取法与 BarDirection 一致 —— boxPoints 的顺序和直觉相反:
    // p0->p1 是 size.height 方向、p1->p2 是 size.width 方向(见 BarDirection 的说明)。
    // 原来这里写成 (size.height >= size.width) ? edge_h : edge_w,两个分支取到的
    // 都是【短边】,于是下面的 along 恒为短边、n 恒为长轴,整个函数的前提就是错的。
    cv::Point2f along = (bar.rect.size.width >= bar.rect.size.height)
                            ? bar.points[2] - bar.points[1]
                            : bar.points[1] - bar.points[0];
    const float alen = std::hypot(along.x, along.y);
    if (alen < 1e-3f) {
        SplitRectCorners(bar);
        return;
    }
    const cv::Point2f u = along / alen;   // 长轴单位向量:沿灯条方向
    const cv::Point2f n(-u.y, u.x);       // 横跨灯条方向(垂直于长轴)

    // 分区必须【横跨长轴】来分两侧,把灯条切成两条长边。
    // 原来这里写的是"取和 x 轴更贴合的那个方向":灯条竖直时那个方向恰好就是
    // 横跨方向,所以 0° 看着是对的;可灯条一旦躺平,它选的变成沿长轴的方向,
    // left_*/right_* 于是取成了灯条的【两个端点】而不是【两条长边】,
    // 四边形错开整整一根灯条 —— 实测 45°~120° 误差 ~390px ≈ 灯条长 410px。
    cv::Point2f dir = n;
    if (dir.x < 0.0f) dir = -dir;         // 统一指向右,left_* 落在 -dir 侧

    // 轮廓点全部投影到 dir 上,两端即极值
    std::vector<float> s(contour.size());
    float smin = 1e9f, smax = -1e9f;
    for (size_t i = 0; i < contour.size(); ++i) {
        s[i] = (cv::Point2f(contour[i]) - bar.center).dot(dir);
        smin = std::min(smin, s[i]);
        smax = std::max(smax, s[i]);
    }

    // band 是"算作同一条边"的容差:太小会把抗锯齿的毛刺排除掉导致端点选偏,
    // 太大则会把对侧的点也拉进来。按灯条厚度取,不再用 length。
    const float band0 = std::max(2.0f, 0.20f * bar.width);
    auto pick_side = [&](bool left_side) {
        std::vector<cv::Point2f> sel;
        // 上限取 0.6*width:两侧之间隔着整整一个 width,band 不到 width 就不会
        // 把对侧的点捞进来;下限给到 3 是为了让细灯条也能多试一档容差
        // (原先上限 0.5*width,7px 宽的灯条只能试 band=2 一次,凑不够两个点就退化)。
        for (float band = band0; band <= std::max(3.0f, 0.6f * bar.width); band *= 2.0f) {
            sel.clear();
            for (size_t i = 0; i < contour.size(); ++i) {
                if (left_side ? (s[i] <= smin + band) : (s[i] >= smax - band)) {
                    sel.emplace_back(contour[i]);
                }
            }
            if (sel.size() >= 2) break;   // 一条边至少要凑出两个端点
        }
        if (sel.size() < 2) return std::make_pair(bar.center, bar.center);

        // 在这条边上取【沿长轴】的两个端点,不能按 y 取:
        // 灯条横躺时这条边是水平的,边上所有点 y 几乎相同,按 y 排序会取到
        // 边中间的两个点,真正的端点反而丢了。
        float tmin = 1e9f, tmax = -1e9f;
        cv::Point2f pa = sel[0], pb = sel[0];
        for (const cv::Point2f &p : sel) {
            const float t = (p - bar.center).dot(u);
            if (t < tmin) { tmin = t; pa = p; }
            if (t > tmax) { tmax = t; pb = p; }
        }
        // 命名仍按图像里的上下:y 小的叫 top,符合"左灯条最左侧上下两点"的说法。
        // 灯条横躺时两个 y 几乎相等,取哪个都不影响四边形本身的正确性。
        return pa.y <= pb.y ? std::make_pair(pa, pb) : std::make_pair(pb, pa);
    };

    auto L = pick_side(true);
    auto R = pick_side(false);
    bar.left_top = L.first;
    bar.left_bottom = L.second;
    bar.right_top = R.first;
    bar.right_bottom = R.second;
}

// 从二值图里提取并筛选灯条
// value_channel 必须是 HSV 的 V 通道(单通道):
//   注意不能用灰度图代替 —— 纯红转灰度只有 76、纯蓝只有 29,
//   拿灰度当"亮度"会把纯色灯条全部误杀。V = max(B,G,R) 才反映实际发光强度。
std::vector<LightBar> DetectLightBars(const cv::Mat &mask, const cv::Mat &value_channel, int color) {
    std::vector<LightBar> bars;

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()) return bars;

    bars.reserve(contours.size());
    for (const auto &contour : contours) {
        // 只要够 minAreaRect 用就行(3 个点)。
        // 这里绝对不能写 <5:CHAIN_APPROX_SIMPLE 会把共线的点压掉,
        // 一个"正对镜头的轴对齐矩形灯条"恰好只剩 4 个角点,
        // 用 <5 判断会把它整根丢掉 —— 正对镜头反而查不到,就是这个原因。
        if (contour.size() < 3) continue;

        LightBar bar;
        bar.rect = cv::minAreaRect(contour);
        bar.length = std::max(bar.rect.size.width, bar.rect.size.height);
        bar.width = std::min(bar.rect.size.width, bar.rect.size.height);

        // 1) 绝对尺寸下限。这是必须的:下面每一道关卡(长宽比、填充率、
        //    角度差、长度差、中心距比、轴偏比)全是【比值】,与尺度无关,
        //    于是两块相距 3px 的 3x2 噪点和两块相距 1000px 的 400x60 灯条
        //    能拿到完全一样的分数。实测拿一张普通图纸(984x733,只有抗锯齿的
        //    文字和色块)去跑,冒出 3x2 / 4x2 / 9x5 / 12x3 的"灯条",
        //    还配出了 score 0.89 的"装甲板"—— 比两张真实装甲板(0.67/0.70)还高。
        //    这里把下限压得很低,只为滤掉零星噪点:实测真实灯条缩到 0.12
        //    (47x8,面积约 350)仍比下限大 5 倍以上,不影响远距离识别。
        double contour_area_raw = cv::contourArea(contour);
        if (contour_area_raw < BarCfg::kMinArea || bar.length < BarCfg::kMinLength) {
            continue;
        }

        // 2) 长宽比:灯条必须是细长条,方块/整块亮区会被滤掉
        float ratio = bar.length / bar.width;
        if (ratio < BarCfg::kMinRatio || ratio > BarCfg::kMaxRatio) {
            continue;
        }

        // 3) 填充率:轮廓要能填满外接矩形的大部分,太散说明是噪声
        double contour_area = cv::contourArea(contour);
        double rect_area = bar.length * bar.width;
        if (rect_area < 1e-3) continue;
        float fill = static_cast<float>(contour_area / rect_area);
        if (fill < BarCfg::kMinFill || fill > BarCfg::kMaxFill) {
            continue;
        }

        bar.points.clear();
        cv::Mat box(4, 2, CV_32F);
        cv::boxPoints(bar.rect, box);
        bar.points.resize(4);
        for (int i = 0; i < 4; ++i) bar.points[i] = box.at<cv::Point2f>(i);
        bar.center = bar.rect.center;

        // 4) 角度:RotatedRect 的 width 方向是 (p0->p1),height 方向是 (p1->p2)
        float long_axis_angle;
        if (bar.rect.size.height >= bar.rect.size.width) {
            long_axis_angle = std::atan2(bar.points[2].y - bar.points[1].y,
                                         bar.points[2].x - bar.points[1].x) * 180.0f / CV_PI;
        } else {
            long_axis_angle = std::atan2(bar.points[1].y - bar.points[0].y,
                                         bar.points[1].x - bar.points[0].x) * 180.0f / CV_PI;
        }
        bar.angle = NormalizeAngle(long_axis_angle);

        // 5) 亮度:只在灯条【自己的像素】上对 V 通道取平均。
        //    这里踩过坑:原先用 bar.rect.boundingRect() 取 ROI 再求均值,但那是旋转矩形的
        //    【轴对齐】外接框 —— 灯条竖直时框贴合,一旦倾斜 30°,框面积涨到灯条本体的
        //    3.6 倍,均值被大片暗背景拉垮:同一根灯条 0° 量到 162、10° 只剩 98、30° 只剩 64,
        //    而 kMinBright=130。结果是"转一点点就检不出",且和真实亮度无关,纯粹是几何假象。
        //    改成把轮廓填成掩膜、只统计掩膜内的像素,量出来的亮度就与旋转无关了。
        cv::Rect roi = bar.rect.boundingRect() & cv::Rect(0, 0, value_channel.cols, value_channel.rows);
        if (roi.area() <= 0) continue;
        {
            cv::Mat roi_mask = cv::Mat::zeros(roi.size(), CV_8UC1);
            std::vector<cv::Point> shifted;
            shifted.reserve(contour.size());
            for (const cv::Point &p : contour) {
                shifted.emplace_back(p.x - roi.x, p.y - roi.y);
            }
            const std::vector<std::vector<cv::Point>> one{shifted};
            cv::drawContours(roi_mask, one, 0, cv::Scalar(255), cv::FILLED);
            if (cv::countNonZero(roi_mask) == 0) continue;
            bar.brightness = static_cast<float>(cv::mean(value_channel(roi), roi_mask)[0]);
        }
        if (bar.brightness < BarCfg::kMinBright) {
            continue;
        }

        // 6) 角点:优先用亮度梯度定边(对曝光/白平衡的缩放不变,蓝色角点就靠它稳住),
        //    找不齐时回落到轮廓极值法,避免出现"没有角点"的断崖
        if (!ExtractCornersByGradient(value_channel, bar)) {
            ExtractOuterCornersByContour(contour, bar);
        }

        bar.color = color;
        bars.push_back(bar);
    }
    return bars;
}

// ==================================================================
// 五、灯条配对 -> 装甲板四角
// ==================================================================

// 灯条长轴的单位方向向量(指向长边方向)。
// 这里踩过坑,必须记下 boxPoints 的【实际】顺序 —— 它和直觉是反的:
// 实测一根 rect.size=(410.2, 69.2) 的灯条,p0->p1 的长度是 69.2(= size.height),
// p1->p2 的长度是 410.2(= size.width)。也就是说 p0->p1 是 height 方向、
// p1->p2 是 width 方向。所以我原先按注释"width 是 p0->p1"取长轴,取到的是【短边】,
// 方向近水平,中心连线往上一投影就是 990px,配对全灭。
//   (同样因此,DetectLightBars 里算出来的 bar.angle 量的是短边方向而非长轴方向。
//    它依然稳定、两根灯条之间的角度差依然可靠,故不改动,只在此备注。)
cv::Point2f BarDirection(const LightBar &bar) {
    // 必须比较 rect 自己的 width/height,不能写 bar.length >= bar.width:
    // length 按定义就是 max(width,height),那个条件恒为真,等于永远取 p2-p1,
    // 只在 size.width 恰好是长边时才碰巧对 —— 实测 z=500/yaw=50° 时 OpenCV
    // 把某根矩形的 width/height 报反了,轴偏瞬间从 0.008 跳到 1.413,配对归零。
    cv::Point2f d = bar.rect.size.width >= bar.rect.size.height ? bar.points[2] - bar.points[1]
                                                                : bar.points[1] - bar.points[0];
    float n = std::hypot(d.x, d.y);
    return n > 1e-6f ? cv::Point2f(d.x / n, d.y / n) : cv::Point2f(0.0f, 1.0f);
}

// 两根灯条是不是同一块装甲板?返回得分(0~1),不匹配返回 0
float MatchScore(const LightBar &a, const LightBar &b, int *large_armor) {
    if (a.color != b.color) return 0.0f;

    // 1) 平行性:两根灯条朝向必须基本一致
    float angle_diff = std::abs(a.angle - b.angle);
    if (angle_diff > PairCfg::kMaxAngleDiff) return 0.0f;

    // 2) 长度接近:同一块装甲板的两根灯条尺寸相近
    float avg_length = (a.length + b.length) * 0.5f;
    float length_diff = std::abs(a.length - b.length) / avg_length;
    if (length_diff > PairCfg::kMaxLengthDiffR) return 0.0f;

    // 3) 中心距 / 灯条长度 ≈ 装甲板高宽比,过小是两灯条挨太近,过大是两个无关目标
    cv::Point2f delta = b.center - a.center;
    float dist = std::hypot(delta.x, delta.y);
    if (dist < PairCfg::kMinCenterDist) return 0.0f;
    float dist_ratio = dist / avg_length;
    if (dist_ratio < PairCfg::kMinDistRatio || dist_ratio > PairCfg::kMaxDistRatio) return 0.0f;

    // 4) 沿长轴方向的错位:同一块装甲板的两根灯条是"肩并肩"的,中心连线应该
    //    几乎垂直于灯条长轴,在长轴方向上的投影分量越小越好。
    //    这里必须投影到灯条自身的坐标轴上,不能直接用图像的 Δy —— 踩过坑:
    //    装甲板绕 roll 转 θ 时 Δy 按 d·sinθ 涨上去,实测 10° 就到 0.585、
    //    30° 到 1.381,而 kMaxYDiffRatio=0.5,于是"歪一点就配不上";
    //    可那两根灯条全程严格平行(角度差恒 0.5°)、相对位置一点没变,
    //    变大的是坐标系转了之后的 Δy,不是错位。投影到长轴后这个量在
    //    roll 0°~40° 全程 ≈ 0,与旋转无关。
    cv::Point2f dir = BarDirection(a);
    float off_axis = std::abs(delta.x * dir.x + delta.y * dir.y);
    if (off_axis / avg_length > PairCfg::kMaxYDiffRatio) return 0.0f;

    *large_armor = dist_ratio < PairCfg::kSmallArmorRatio ? 1 : 0;

    // 打分:越平行、长度越接近、比例越合理分越高
    float s_angle = 1.0f - angle_diff / PairCfg::kMaxAngleDiff;
    float s_len = 1.0f - length_diff / PairCfg::kMaxLengthDiffR;
    // 高宽比以"大装甲 ~1.2"附近为最优,越小越偏小装甲
    float ideal_ratio = *large_armor ? 1.2f : PairCfg::kSmallArmorRatio + 1.0f;
    float s_dist = 1.0f - std::abs(dist_ratio - ideal_ratio) / ideal_ratio;
    s_dist = std::max(0.0f, s_dist);
    float s_y = 1.0f - (off_axis / avg_length) / PairCfg::kMaxYDiffRatio;

    float score = PairCfg::kWeightAngle * s_angle + PairCfg::kWeightLength * s_len +
                  PairCfg::kWeightDist * s_dist + PairCfg::kWeightY * s_y;
    return std::min(1.0f, std::max(0.0f, score));
}

std::vector<ArmorPlate> FindArmors(const std::vector<LightBar> &bars) {
    std::vector<ArmorPlate> armors;
    if (bars.size() < 2) return armors;

    for (size_t i = 0; i < bars.size(); ++i) {
        for (size_t j = i + 1; j < bars.size(); ++j) {
            int large_armor = 1;
            float score = MatchScore(bars[i], bars[j], &large_armor);
            if (score < PairCfg::kScoreThreshold) continue;

            // 谁在左谁在右:按中心 x 排
            const LightBar *lb = &bars[i];
            const LightBar *rb = &bars[j];
            if (lb->center.x > rb->center.x) std::swap(lb, rb);

            ArmorPlate armor;
            armor.color = lb->color;
            armor.score = score;
            armor.left = *lb;
            armor.right = *rb;
            // 脚点:左灯条的外沿 + 右灯条的外沿。
            // 装甲板绕 yaw 转起来时两边投影长度不等,这四个点连出来就是梯形,
            // 不需要为梯形做特判 —— 取"最外侧"本身就是梯形的四条边。
            armor.corners = {lb->left_top, rb->right_top, rb->right_bottom, lb->left_bottom};

            // 梯形左右两条边的长度,以及它们的比值。
            // 正对镜头时两条边等长 trape_ratio = 1.0;
            // 绕 yaw 转开后远端灯条投影变短,比值随之变小 —— 可以直接当偏转角指标用。
            armor.left_edge = cv::norm(armor.corners[3] - armor.corners[0]);
            armor.right_edge = cv::norm(armor.corners[2] - armor.corners[1]);
            const float long_edge = std::max(armor.left_edge, armor.right_edge);
            armor.trape_ratio = long_edge > 1e-3f ? std::min(armor.left_edge, armor.right_edge) / long_edge
                                                  : 1.0f;

            // 中心取四个角点的形心:梯形情况下比"两灯条中心的中点"更贴近视觉中心
            armor.center = (armor.corners[0] + armor.corners[1] + armor.corners[2] + armor.corners[3]) * 0.25f;
            armors.push_back(armor);
        }
    }

    // 得分高的排前面
    std::sort(armors.begin(), armors.end(),
              [](const ArmorPlate &a, const ArmorPlate &b) { return a.score > b.score; });
    return armors;
}

// 跨帧稳定编号:与上一帧中心最近的装甲板沿用同一个 id
void AssignIds(std::vector<ArmorPlate> &armors, const std::vector<ArmorPlate> &prev) {
    std::vector<bool> used(prev.size(), false);
    for (auto &armor : armors) {
        int best = -1;
        float best_dist = 1e9f;
        for (size_t k = 0; k < prev.size(); ++k) {
            if (used[k] || prev[k].color != armor.color) continue;
            float d = cv::norm(armor.center - prev[k].center);
            if (d < best_dist) {
                best_dist = d;
                best = static_cast<int>(k);
            }
        }
        if (best >= 0 && best_dist < 80.0f) {
            used[best] = true;
            armor.id = prev[best].id;
        } else {
            static int next_id = 0;
            armor.id = next_id++;
        }
    }
}

// ==================================================================
// 六、绘制
// ==================================================================

// OpenCV 的 polylines 只吃 CV_32S 的点,浮点角点要先取整
static std::vector<cv::Point> ToIntPoints(const std::vector<cv::Point2f> &pts) {
    std::vector<cv::Point> out;
    out.reserve(pts.size());
    for (const auto &p : pts) out.emplace_back(cvRound(p.x), cvRound(p.y));
    return out;
}

void DrawArmors(cv::Mat &image, const std::vector<LightBar> &bars,
                const std::vector<ArmorPlate> &armors) {
    // 1) 所有灯条候选(细线),包括没配对上的
    for (const auto &bar : bars) {
        cv::polylines(image, {ToIntPoints(bar.points)}, true, ColorDraw(bar.color), 1, cv::LINE_AA);
        cv::circle(image, bar.center, 3, ColorDraw(bar.color), -1);
    }

    // 2) 配对成功的装甲板:外框 + 四个脚点 + 中心
    for (const auto &armor : armors) {
        cv::Scalar c = ColorDraw(armor.color);

        // 左右灯条连线
        cv::line(image, armor.left.center, armor.right.center, cv::Scalar(200, 200, 200), 1,
                 cv::LINE_AA);

        // 装甲板外框(四个脚点连成的四边形)
        cv::polylines(image, {ToIntPoints(armor.corners)}, true, c, 2, cv::LINE_AA);

        // 四个脚点
        for (size_t k = 0; k < armor.corners.size(); ++k) {
            cv::circle(image, armor.corners[k], 4, c, -1, cv::LINE_AA);
            cv::circle(image, armor.corners[k], 4, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        }

        // 几何中心十字
        cv::circle(image, armor.center, 3, cv::Scalar(0, 255, 255), -1, cv::LINE_AA);
        cv::line(image, armor.center - cv::Point2f(12, 0), armor.center + cv::Point2f(12, 0),
                 cv::Scalar(0, 255, 255), 1);
        cv::line(image, armor.center - cv::Point2f(0, 12), armor.center + cv::Point2f(0, 12),
                 cv::Scalar(0, 255, 255), 1);

        // 标签:颜色 + id + 得分 + 尺寸,两行都画在框的上方,避免压住外框
        cv::Point2f top = armor.corners[0];
        for (const auto &p : armor.corners) top.y = std::min(top.y, p.y);
        const int size_y = std::max(34, static_cast<int>(top.y) - 8);
        const int label_y = size_y - 20;
        const int label_x = static_cast<int>(armor.center.x) - 40;

        char label[64];
        std::snprintf(label, sizeof(label), "%s #%d  %.2f", ColorName(armor.color), armor.id,
                      armor.score);
        cv::putText(image, label, cv::Point(label_x, label_y), cv::FONT_HERSHEY_SIMPLEX, 0.6, c, 2);

        // 梯形比 = 短边/长边,1.00 表示正对镜头,越小说明绕 yaw 转得越多。
        // 直接画在两条竖边上,方便一眼看出当前是不是梯形。
        char size_text[64];
        std::snprintf(size_text, sizeof(size_text), "h=%.0f/%.0f  yaw=%.2f", armor.left_edge,
                      armor.right_edge, armor.trape_ratio);
        cv::putText(image, size_text, cv::Point(label_x, size_y), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    c, 1);
    }
}

void DrawHud(cv::Mat &image, double fps, double cost_ms, int armor_count, int bar_count,
             const std::string &extra) {
    char fps_text[64];
    std::snprintf(fps_text, sizeof(fps_text), "FPS: %.1f  Cost: %.2f ms", fps, cost_ms);
    cv::putText(image, fps_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                cv::Scalar(0, 255, 0), 2);
    cv::putText(image, "Armors: " + std::to_string(armor_count) +
                       "  LightBars: " + std::to_string(bar_count),
               cv::Point(10, 58), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
    if (!extra.empty()) {
        cv::putText(image, extra, cv::Point(10, image.rows - 15), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(180, 180, 180), 1);
    }
}

// ==================================================================
// 七、main
// ==================================================================

int main(int argc, char *argv[]) {
    spdlog::set_pattern("[%H:%M:%S.%e] [%^%l%$] %v");
    spdlog::set_level(spdlog::level::info);

    // ------------------------------------------------------------------
    // 1. 初始化库 + 枚举设备 + 选设备
    // ------------------------------------------------------------------
    GX_STATUS emStatus = GXInitLib();
    if (emStatus != GX_STATUS_SUCCESS) {
        spdlog::error("GXInitLib 失败: {}", GetErrorString(emStatus));
        return -1;
    }

    auto devices = EnumerateDevices();
    if (devices.empty()) {
        GXCloseLib();
        return 1;
    }

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

    // ------------------------------------------------------------------
    // 2. 打开设备并配置
    // ------------------------------------------------------------------
    GX_DEV_HANDLE device = nullptr;
    emStatus = GXOpenDeviceByIndex(device_index, &device);
    if (emStatus != GX_STATUS_SUCCESS) {
        spdlog::error("GXOpenDeviceByIndex: {}", GetErrorString(emStatus));
        GXCloseLib();
        return -1;
    }
    spdlog::info("打开相机 {} 成功", serial);

    GXSetEnumValueByString(device, "ExposureAuto", "Off");
    GXSetEnumValueByString(device, "GainAuto", "Off");
    // 白平衡必须锁死,原来是 "Continuous" —— 相机每帧重解一次白平衡,
    // 而它动的正是 B 通道增益。蓝色灯条的掩膜外沿恰恰是由 S 阈值决定的
    // (实测:蓝色掩膜边界被 S 砍掉灯条自身 24%~36% 的像素;红色由 H 窗决定,
    // 自己一个像素都不掉),而 S 正是 B 增益直接改变的量。于是蓝色灯条的轮廓
    // 随白平衡"呼吸",角点跟着动。这条管道没有任何一处需要每帧重解白平衡。
    // 先试 Once(开机解一次然后锁住,颜色不至于跑偏),不支持再退 Off。
    if (GXSetEnumValueByString(device, "BalanceWhiteAuto", "Once") != GX_STATUS_SUCCESS) {
        if (GXSetEnumValueByString(device, "BalanceWhiteAuto", "Off") != GX_STATUS_SUCCESS) {
            spdlog::error("锁定白平衡失败,白平衡仍在自动状态,蓝色角点会漂");
        } else {
            spdlog::info("白平衡 -> Off(这台相机不支持 Once)");
        }
    } else {
        spdlog::info("白平衡 -> Once(开机解一次后锁住)");
    }
    GXSetEnumValueByString(device, "AcquisitionMode", "Continuous");
    GXSetEnumValueByString(device, "TriggerMode", "Off");
    GXSetEnumValue(device, "PixelFormat", GX_PIXEL_FORMAT_BAYER_RG8);

    // LED 灯条很亮,曝光拉低一点避免过曝糊成一块
    GXSetFloatValue(device, "ExposureTime", 3000.0);

    GX_INT_VALUE limit_node;
    memset(&limit_node, 0, sizeof(GX_INT_VALUE));
    if (GXGetIntValue(device, "DeviceLinkThroughputLimit", &limit_node) == GX_STATUS_SUCCESS) {
        GXSetIntValue(device, "DeviceLinkThroughputLimit", limit_node.nMax);
    }

    GX_INT_VALUE width_node, height_node;
    memset(&width_node, 0, sizeof(GX_INT_VALUE));
    memset(&height_node, 0, sizeof(GX_INT_VALUE));
    GXGetIntValue(device, "Width", &width_node);
    GXGetIntValue(device, "Height", &height_node);
    const int width = static_cast<int>(width_node.nCurValue);
    const int height = static_cast<int>(height_node.nCurValue);
    spdlog::info("分辨率 {}x{}", width, height);

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

    GXSetAcqusitionBufferNumber(device, 5);
    const unsigned int buffer_size = sizeof(unsigned char) * width * height * 3;
    unsigned char *rgb_buffer = static_cast<unsigned char *>(malloc(buffer_size));
    if (rgb_buffer == nullptr) {
        spdlog::error("分配 rgb 缓冲区失败");
        GXCloseDevice(device);
        GXCloseLib();
        return -1;
    }

    int64_t color_filter = GX_COLOR_FILTER_NONE;
    GX_ENUM_VALUE filter_value;
    memset(&filter_value, 0, sizeof(GX_ENUM_VALUE));
    if (GXGetEnumValue(device, "PixelColorFilter", &filter_value) == GX_STATUS_SUCCESS) {
        color_filter = filter_value.stCurValue.nCurValue;
    }

    emStatus = GXStreamOn(device);
    if (emStatus != GX_STATUS_SUCCESS) {
        spdlog::error("GXStreamOn: {}", GetErrorString(emStatus));
        free(rgb_buffer);
        GXCloseDevice(device);
        GXCloseLib();
        return -1;
    }

    // ------------------------------------------------------------------
    // 3. 识别循环
    // ------------------------------------------------------------------
    spdlog::info("装甲板识别启动:b 显示二值图,p 量测灯条 HSV,e/d 调曝光,a/q 调增益,ESC 退出");

    cv::Mat display_mask;                       // b 键打开二值图窗口
    bool show_mask = false;
    std::vector<ArmorPlate> prev_armors;  // 上一帧结果,用于稳定编号

    int frame_count = 0;
    auto fps_window_start = std::chrono::steady_clock::now();
    double fps = 0.0;
    double cost_ms = 0.0;

    PGX_FRAME_BUFFER frame_buffer = nullptr;
    while (true) {
        emStatus = GXDQBuf(device, &frame_buffer, 1000);
        if (emStatus != GX_STATUS_SUCCESS) {
            spdlog::error("GXDQBuf 超时/失败: {}", GetErrorString(emStatus));
            break;
        }

        if (frame_buffer->nStatus != GX_FRAME_STATUS_SUCCESS) {
            spdlog::error("帧状态异常: {:#x}", static_cast<uint32_t>(frame_buffer->nStatus));
            GXQBuf(device, frame_buffer);
            continue;
        }

        VxInt32 dx_status = DxRaw8toRGB24Ex(
            frame_buffer->pImgBuf, rgb_buffer, frame_buffer->nWidth, frame_buffer->nHeight,
            RAW2RGB_NEIGHBOUR, DX_PIXEL_COLOR_FILTER(color_filter), false, DX_ORDER_BGR);
        if (dx_status != DX_OK) {
            spdlog::error("DxRaw8toRGB24Ex 失败: {:#x}", static_cast<uint32_t>(dx_status));
            GXQBuf(device, frame_buffer);
            continue;
        }

        cv::Mat bgr(frame_buffer->nHeight, frame_buffer->nWidth, CV_8UC3, rgb_buffer);

        auto algo_begin = std::chrono::steady_clock::now();

        // ---- 识别:HSV 阈值 -> 灯条 -> 配对 ----
        cv::Mat hsv;
        cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);

        // V 通道单独取出来给亮度筛选用(extractChannel 只是取视图,不复制数据)
        cv::Mat value_channel;
        cv::extractChannel(hsv, value_channel, 2);

        std::vector<LightBar> bars;
        std::vector<cv::Mat> masks;
        for (int color : {kRed, kBlue}) {
            cv::Mat mask = BuildColorMask(hsv, color);
            if (show_mask) masks.push_back(mask.clone());
            std::vector<LightBar> found = DetectLightBars(mask, value_channel, color);
            bars.insert(bars.end(), found.begin(), found.end());
        }

        std::vector<ArmorPlate> armors = FindArmors(bars);
        AssignIds(armors, prev_armors);
        prev_armors = armors;

        cost_ms = std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - algo_begin).count();

        // ---- 帧率统计 ----
        ++frame_count;
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - fps_window_start).count();
        if (elapsed >= 1.0) {
            fps = frame_count / elapsed;
            frame_count = 0;
            fps_window_start = now;

            // 控制台输出识别帧率 / 耗时 / 识别到的目标
            std::string detail;
            for (const auto &armor : armors) {
                char buf[64];
                std::snprintf(buf, sizeof(buf), " [%s#%d score=%.2f 梯形比=%.2f]", ColorName(armor.color),
                              armor.id, armor.score, armor.trape_ratio);
                detail += buf;
            }
            // 用 debug 级:默认级别是 info,所以这行不再进控制台;
            // 同样的数据窗口 HUD 上已经有了(DrawHud)。
            spdlog::debug("识别帧率: {:.1f} FPS | 耗时: {:.2f} ms | 灯条: {} | 装甲板: {}{}",
                          fps, cost_ms, bars.size(), armors.size(), detail);
        }

        // ---- 显示 ----
        cv::Mat display = bgr.clone();
        DrawArmors(display, bars, armors);
        DrawHud(display, fps, cost_ms, static_cast<int>(armors.size()), static_cast<int>(bars.size()),
                "b: show mask  e/d: exposure  a/q: gain  ESC: quit");
        cv::imshow("armor_recognize", display);

        if (show_mask && !masks.empty()) {
            cv::Mat merged;
            if (masks.size() == 1) {
                merged = masks[0];
            } else {
                merged = masks[0].clone();
                for (size_t k = 1; k < masks.size(); ++k) cv::bitwise_or(merged, masks[k], merged);
            }
            display_mask = merged;
            cv::imshow("binary_mask", display_mask);
        }

        bool quit = false;
        switch (cv::waitKey(1)) {
            case 27: quit = true; break;                 // ESC
            case 'b': show_mask = !show_mask;
                if (!show_mask) cv::destroyWindow("binary_mask");
                spdlog::info("二值图窗口: {}", show_mask ? "显示" : "关闭");
                break;
            case 'e': AddExposureTime(device, 250.0); break;
            case 'd': AddExposureTime(device, -250.0); break;
            case 'a': AddGain(device, 0.1); break;
            case 'q': AddGain(device, -0.1); break;
            case 'p': DumpBrightHue(hsv); break;         // 量测灯条的 H/S/V
            default: break;
        }

        GXQBuf(device, frame_buffer);
        if (quit) break;
    }

    GXStreamOff(device);
    free(rgb_buffer);
    GXCloseDevice(device);
    GXCloseLib();
    spdlog::info("相机已关闭");
    return 0;
}