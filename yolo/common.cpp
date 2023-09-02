#include "pch.hpp"
#include "common.hpp"

const unsigned char colors[81][3] = {
    {56,  0,   255},
    {226, 255, 0},
    {0,   94,  255},
    {0,   37,  255},
    {0,   255, 94},
    {255, 226, 0},
    {0,   18,  255},
    {255, 151, 0},
    {170, 0,   255},
    {0,   255, 56},
    {255, 0,   75},
    {0,   75,  255},
    {0,   255, 169},
    {255, 0,   207},
    {75,  255, 0},
    {207, 0,   255},
    {37,  0,   255},
    {0,   207, 255},
    {94,  0,   255},
    {0,   255, 113},
    {255, 18,  0},
    {255, 0,   56},
    {18,  0,   255},
    {0,   255, 226},
    {170, 255, 0},
    {255, 0,   245},
    {151, 255, 0},
    {132, 255, 0},
    {75,  0,   255},
    {151, 0,   255},
    {0,   151, 255},
    {132, 0,   255},
    {0,   255, 245},
    {255, 132, 0},
    {226, 0,   255},
    {255, 37,  0},
    {207, 255, 0},
    {0,   255, 207},
    {94,  255, 0},
    {0,   226, 255},
    {56,  255, 0},
    {255, 94,  0},
    {255, 113, 0},
    {0,   132, 255},
    {255, 0,   132},
    {255, 170, 0},
    {255, 0,   188},
    {113, 255, 0},
    {245, 0,   255},
    {113, 0,   255},
    {255, 188, 0},
    {0,   113, 255},
    {255, 0,   0},
    {0,   56,  255},
    {255, 0,   113},
    {0,   255, 188},
    {255, 0,   94},
    {255, 0,   18},
    {18,  255, 0},
    {0,   255, 132},
    {0,   188, 255},
    {0,   245, 255},
    {0,   169, 255},
    {37,  255, 0},
    {255, 0,   151},
    {188, 0,   255},
    {0,   255, 37},
    {0,   255, 0},
    {255, 0,   170},
    {255, 0,   37},
    {255, 75,  0},
    {0,   0,   255},
    {255, 207, 0},
    {255, 0,   226},
    {255, 245, 0},
    {188, 255, 0},
    {0,   255, 18},
    {0,   255, 75},
    {0,   255, 151},
    {255, 56,  0},
    {245, 255, 0}
};

std::vector<std::string> IMG_FORMATS{ "bmp", "dng", "jpg", "jpeg", "mpo", "png", "tif", "tiff", "webp", "pfm" };
std::vector<std::string> VID_FORMATS{ "asf", "avi", "gif", "m4v", "mkv", "mov", "mp4", "mpeg", "mpg", "ts", "wmv" };

Timer::Timer(const char* _task){
    task = _task;
    start = std::chrono::high_resolution_clock::now();
}

Timer::~Timer() {
    finish = std::chrono::high_resolution_clock::now();
    duration = finish - start;

    std::cout << task << " took " << duration << std::endl;
}

bool isImage(const std::string& path) {
	std::string ext = path.substr(path.find_last_of(".") + 1);
	std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });
	return std::find(IMG_FORMATS.begin(), IMG_FORMATS.end(), ext) != IMG_FORMATS.end();
}

bool isImage(const std::filesystem::path& path) {
    std::string ext = path.extension().string().substr(1);
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });
    return std::find(IMG_FORMATS.begin(), IMG_FORMATS.end(), ext) != IMG_FORMATS.end();
}

bool isVideo(const std::string& path) {
	std::string ext = path.substr(path.find_last_of(".") + 1);
	std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });
	return std::find(VID_FORMATS.begin(), VID_FORMATS.end(), ext) != VID_FORMATS.end();
}

bool isVideo(const std::filesystem::path& path) {
    std::string ext = path.extension().string().substr(1);
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });
    return std::find(VID_FORMATS.begin(), VID_FORMATS.end(), ext) != VID_FORMATS.end();
}