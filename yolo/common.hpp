#pragma once

struct Timer {
    std::chrono::time_point<std::chrono::steady_clock> start{}, finish{};
    std::chrono::duration<float> duration{}; //duration in seconds
    std::string task;
    Timer(const char* _task);
    ~Timer();
};

extern const unsigned char colors[81][3];

extern std::vector<std::string> IMG_FORMATS;
extern std::vector<std::string> VID_FORMATS;

bool isImage(const std::string& path);

bool isImage(const std::filesystem::path& path);

bool isVideo(const std::string& path);

bool isVideo(const std::filesystem::path& path);