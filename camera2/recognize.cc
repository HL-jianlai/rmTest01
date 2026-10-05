// 大恒(Galaxy)工业相机 + OpenCV 装甲板(armor)识别
//
// 思路(BGR/HSV 双通道判定灯条颜色):
//   1. 取流 -> Bayer RG8 转 BGR24,拿到 OpenCV 的 cv::Mat
//   2. HSV 通道做颜色阈值,红色灯条要拆成两段色相区间([0,10] ∪ [160,180])
//      蓝色灯条落在 [100,130];S/V 双阈值用来压掉暗部和低饱和背景
//   3. 形态学闭运算把 LED 灯珠连成整根灯条,再开运算去噪点
//   4. findContours + minAreaRect,用 长宽比 / 填充率 / 角度 / 亮度 筛出灯条候选
//      (刻意不限制灯条的像素大小,远近目标一视同仁)
//   5. 同色灯条两两配对:中心距、灯条长度比、朝向夹角、位置关系打分
//   6. 由左灯条左边缘 + 右灯条右边缘拼出装甲板四角(脚点),画框画点
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
namespace ColorCfg {
constexpr int kRedHMin1   = 0;    // 红:色相环绕,拆两段
constexpr int kRedHMax1   = 10;
constexpr int kRedHMin2   = 160;
constexpr int kRedHMax2   = 179;
constexpr int kRedSMin    = 120;   // 饱和度下限,压掉灰白背景
constexpr int kRedVMin    = 100; // 明度下限,压掉暗部

constexpr int kBlueHMin   = 120;
constexpr int kBlueHMax   = 135;
constexpr int kBlueSMin   = 170;
constexpr int kBlueVMin   = 250;
}  // namespace ColorCfg

// ---- 灯条几何筛选 ----
// 注意:这里刻意不设"灯条多长/多宽"的上下限,像素尺寸不限制,
// 只靠"长宽比 + 填充率 + 亮度"判断,远近大小的灯条都能进得来。
namespace BarCfg {
constexpr float kMinRatio  = 1.5f;    // 长/宽 比下限(灯条是细长条)
constexpr float kMaxRatio  = 18.0f;   // 长/宽 比上限
constexpr float kMinBright = 130.0;// ROI 平均 V,LED 一定比背景亮
constexpr float kMinFill   = 0.25f;   // contour 面积 / minAreaRect 面积
// 实心灯条的填充率天然接近 1.0,这里不设上限,只保留填充率下限。
// 不要往下调这个值:抗锯齿边缘 + 噪声会让 mask 比 minAreaRect 略小,
// 实测真实灯条填充率就在 0.89~0.90 之间,写 0.90 会正好卡在边界上,
// 时检出时漏检,而且长宽比越接近直角越容易踩到。
constexpr float kMaxFill   = 1.00f;
}  // namespace BarCfg

// ---- 灯条配对(装甲板) ----
namespace PairCfg {
constexpr float kMaxAngleDiff     = 20.0f;  // 两灯条长轴夹角上限(度)
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
    double value = node.dCurValue + delta_us;
    value = value < 1.0 ? 1.0 : (value > 10000.0 ? 10000.0 : value);
    emStatus = GXSetFloatValue(device, "ExposureTime", value);
    if (emStatus != GX_STATUS_SUCCESS) {
        spdlog::error("Set ExposureTime: {}", GetErrorString(emStatus));
        return false;
    }
    spdlog::info("曝光时间 -> {} us", value);
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
    value = value < 0.0 ? 0.0 : (value > 32.0 ? 32.0 : value);
    emStatus = GXSetFloatValue(device, "Gain", value);
    if (emStatus != GX_STATUS_SUCCESS) {
        spdlog::error("Set Gain: {}", GetErrorString(emStatus));
        return false;
    }
    spdlog::info("增益 -> {} dB", value);
    return true;
}

// ==================================================================
// 四、灯条检测
// ==================================================================

// 生成某一颜色的二值图。红色要合并两段色相区间(0 附近环绕)。
cv::Mat BuildColorMask(const cv::Mat &hsv, int color, bool show) {
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

    if (show) {
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 9));
        // 竖直方向长核:把断开的 LED 灯珠连成一根完整灯条
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
    }
    return mask;
}

// 把长轴角度归一化到 (-90, 90],方便比较两根灯条是否平行
float NormalizeAngle(float deg) {
    while (deg > 90.0f) deg -= 180.0f;
    while (deg <= -90.0f) deg += 180.0f;
    return deg;
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
        if (bar.length < 1.0f) {
            continue;
        }

        // 1) 不限制灯条像素大小,远近目标一视同仁;
        //    唯一的下限是上面 length >= 1.0,避免下面算长宽比时除零

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

        // 5) 亮度:ROI 内平均 V(HSV 明度通道),LED 条显著高于周围
        cv::Rect roi = bar.rect.boundingRect() & cv::Rect(0, 0, value_channel.cols, value_channel.rows);
        if (roi.area() <= 0) continue;
        bar.brightness = static_cast<float>(cv::mean(value_channel(roi))[0]);
        if (bar.brightness < BarCfg::kMinBright) {
            continue;
        }

        // 6) 拆分左右两侧端点(按 x 排,前两个是最左的)
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

        bar.color = color;
        bars.push_back(bar);
    }
    return bars;
}

// ==================================================================
// 五、灯条配对 -> 装甲板四角
// ==================================================================

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

    // 4) 垂直错位:灯条基本在同一水平线上
    float y_diff = std::abs(delta.y);
    if (y_diff / avg_length > PairCfg::kMaxYDiffRatio) return 0.0f;

    *large_armor = dist_ratio < PairCfg::kSmallArmorRatio ? 1 : 0;

    // 打分:越平行、长度越接近、比例越合理分越高
    float s_angle = 1.0f - angle_diff / PairCfg::kMaxAngleDiff;
    float s_len = 1.0f - length_diff / PairCfg::kMaxLengthDiffR;
    // 高宽比以"大装甲 ~1.2"附近为最优,越小越偏小装甲
    float ideal_ratio = *large_armor ? 1.2f : PairCfg::kSmallArmorRatio + 1.0f;
    float s_dist = 1.0f - std::abs(dist_ratio - ideal_ratio) / ideal_ratio;
    s_dist = std::max(0.0f, s_dist);
    float s_y = 1.0f - (y_diff / avg_length) / PairCfg::kMaxYDiffRatio;

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
    GXSetEnumValueByString(device, "BalanceWhiteAuto", "Continuous");
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
    spdlog::info("装甲板识别启动:b 显示二值图,e/d 调曝光,a/q 调增益,ESC 退出");

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
            cv::Mat mask = BuildColorMask(hsv, color, show_mask);
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
            spdlog::info("识别帧率: {:.1f} FPS | 耗时: {:.2f} ms | 灯条: {} | 装甲板: {}{}",
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
                break;
            case 'e': AddExposureTime(device, 250.0); break;
            case 'd': AddExposureTime(device, -250.0); break;
            case 'a': AddGain(device, 0.1); break;
            case 'q': AddGain(device, -0.1); break;
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