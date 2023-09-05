#pragma once
#include <chrono>

struct Timer {
    std::chrono::time_point<std::chrono::steady_clock> start{}, finish{};
    std::chrono::duration<float> duration{}; //duration in seconds
    std::string task;
    Timer(const char* _task);
    ~Timer();
};