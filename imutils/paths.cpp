#include "pch.hpp"
#include "paths.hpp"

namespace fs = std::filesystem;

/**
* @var imageTypes
* @brief contains all types of image extensions supported by Opencv
**/
std::vector<std::string> imageTypes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"};

std::vector<std::string> imutils::listImages(std::string basePath, std::string contains) {
    return listFiles(basePath, imageTypes, contains = contains);
}

std::vector<std::string> imutils::listFiles(std::string basePath, std::vector<std::string> validExts, std::string contains) {
    std::vector<std::string> filesDirs;
    std::filesystem::path path = basePath;
    for (const auto& dirEntry : fs::recursive_directory_iterator(path)) {
        std::filesystem::path file = dirEntry.path();
        if (contains != "" and (file.filename().string()).find(contains) == std::string::npos)
            continue;
        if (validExts.empty())
            filesDirs.push_back(dirEntry.path().string());
        else if (std::find(validExts.begin(), validExts.end(), file.extension()) != validExts.end())
                filesDirs.push_back(dirEntry.path().string());
    }
    return filesDirs;
}