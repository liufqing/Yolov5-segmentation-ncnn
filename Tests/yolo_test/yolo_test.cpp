#include "utils.hpp"
#include "yolo.hpp"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <filesystem>
#include <opencv2/core/utility.hpp>

int main(int argc, char** argv) {

    std::vector<Object> objs;
    Utils utils(argc, argv);

    utils.run();

    //cv::Mat img = cv::imread("../input/test.jpg");
    //utils.load("../models/yolov5s-seg-idcard-2.ncnn.bin", "../models/yolov5s-seg-idcard-2.ncnn.param");
    //utils.image("./input/test.jpg");

    return 0;
}