#pragma once
#include <string>
#include <vector>
#include <chrono>
#include <filesystem>

#include <opencv2/imgproc/imgproc.hpp>

struct Timer {
    std::chrono::time_point<std::chrono::steady_clock> start{}, finish{};
    std::chrono::duration<float> duration{}; //duration in seconds
    std::string task;
    Timer(const char* _task);
    ~Timer();
};

struct Object {
    cv::Rect_<float> rect;
    int label {};
    float prob {};
    std::vector<float> mask_feat;
    cv::Mat cv_mask;
};

extern const unsigned char colors[81][3];

extern std::vector<std::string> IMG_FORMATS;
extern std::vector<std::string> VID_FORMATS;

bool isImage(const std::string& path);

bool isImage(const std::filesystem::path& path);

bool isVideo(const std::string& path);

bool isVideo(const std::filesystem::path& path);