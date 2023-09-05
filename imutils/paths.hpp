#pragma once
#define PATH_H

#include <filesystem>

namespace imutils {
    /**
    * @param basePath the directory where you want to search
    * @param contains Text that the filename must contain leave empty if none
    * @return Vector of Strings containing relative path of the files
    * @brief Utility for listing all the Images in directory
    **/
    std::vector<std::string> listImages(std::string basePath, std::string contains = "");

    /**
    * @param basePath The directory where you want to search
    * @param validExts List of file extensions
    * @param contains Text that the filename must contain leave empty if none
    * @return Vector of Strings containing relative path of the files
    * @brief Utility for listing all the Files in directory with valid extensions and containing words if any
    **/
    std::vector<std::string> listFiles(std::string basePath, std::vector<std::string> validExts = {}, std::string contains = "");
};