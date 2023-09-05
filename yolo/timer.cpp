#include "pch.hpp"
#include "timer.hpp"

Timer::Timer(const char* _task){
    task = _task;
    start = std::chrono::high_resolution_clock::now();
}

Timer::~Timer() {
    finish = std::chrono::high_resolution_clock::now();
    duration = finish - start;

    std::cout << task << " took " << duration << std::endl;
}