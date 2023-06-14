#pragma once
#include <ncnn/net.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/utils/filesystem.hpp>

#include <iostream>
#include <stdio.h>
#include <string>
#include <float.h>
#include <vector>
#include <fstream>
#include <time.h>
#include <cmath>
#include <filesystem>
#include <windows.h>

#include "common.hpp"

enum strategy {
    concatenatedContour = 0,    //concatenate all segments
    largestContour      = 1     //select largest segment
};

enum colorMode {
    byClass = 0,                //color object by class index
    byIndex = 1                 //color object by object number index 
};

class Yolo {
public:
    Yolo();
    ~Yolo();
    int load(const std::string& bin, const std::string& param);

    int detect(const cv::Mat& bgr, std::vector<Object>& objects);

    int detect_dynamic(const cv::Mat& bgr, std::vector<Object>& objects);

    void draw_mask(cv::Mat& bgr, cv::Mat mask, const unsigned char* color);

    void crop_object(cv::Mat &bgr, cv::Mat mask, cv::Rect rect);

    std::vector<cv::Point> mask2segment(cv::Mat& mask, int strategy = largestContour);

    /**
     * @brief 
     * 
     * @param bgr : background image to be draw on
     * @param objects : object vector contain all the detected object in the image
     * @param colorMode : determine the color for each object to be draw ( bounding box and feature mask )
     */
    cv::Mat draw_objects(cv::Mat bgr, const std::vector<Object>& objects, int colorMode = byIndex);

    void video(cv::VideoCapture capture);

    void image(std::string inputPath);

    void get_class_names(std::string data);

    void get_blob_name(std::string in, std::string out, std::string out0, std::string out1, std::string out2,std::string seg);

    bool dynamic         = false;
    bool save            = false;
    bool noseg           = false;
    bool agnostic        = false;
    bool crop            = false;
    bool saveTxt         = false;
    bool saveMask         = false;
    int target_size      = 640;
    float prob_threshold = 0.25f;
    float nms_threshold  = 0.45f;
    int max_object       = 100;
    std::string outputFolder;
private:
    ncnn::Net net;
    std::vector<std::string> class_names;
    std::vector<Object> objects;
    int class_count=0;
    double inference_time;
    std::string in_blob;
    std::string out_blob;
    std::string out1_blob;
    std::string out2_blob;
    std::string out3_blob;
    std::string seg_blob;
    std::string inputNameWithoutExt;
};