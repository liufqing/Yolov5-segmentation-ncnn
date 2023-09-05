#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>

#include "imutils.hpp"

using std::cout;
using std::endl;

int main(int argc, char* argv[]) {
    const char* input = argv[1];
    const char* minAngle = argv[2];
    const char* maxAngle = argv[3];
    const char* stepAngle = argv[4];

    int min = std::stoi(minAngle);
    int max = std::stoi(maxAngle);
    int step = std::stoi(stepAngle);

    cv::Mat src = cv::imread(input);

    std::filesystem::path inputPath = input;

    cv::Mat dst;
    cv::Size size(src.cols, src.cols);
    std::string outputFolder = inputPath.parent_path().string() + "\\" + inputPath.stem().string();
    cv::utils::fs::createDirectory(outputFolder);
    for (int i = min; i <= max; i += step) {
        dst = imutils::rotateBounds(src, i);
        cout << "Generating " << std::to_string(i) + ".jpg" << endl;
        cv::imwrite(outputFolder + "/" + std::to_string(i) + ".jpg", dst);
    }
    cout << (max - min) / step + 1 << " images Generated are save to " << outputFolder << endl;
    cout << "------------------------------------------------" << std::endl;

    return 0;
}