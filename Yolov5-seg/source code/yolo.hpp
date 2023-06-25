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

#define _USE_MATH_DEFINES
#include <math.h>

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

    int load(const std::filesystem::path& bin, const std::filesystem::path& param);

    int detect(const cv::Mat& bgr, std::vector<Object>& objects);

    int detect_dynamic(const cv::Mat& bgr, std::vector<Object>& objects);

    void video(std::string inputPath);

    void image(const std::filesystem::path& inputPath, const std::filesystem::path& outputFolder, bool continuous = false);

    void get_class_names(const std::string& data);

    void get_class_names(const std::filesystem::path& data);

    void get_blob_name(std::string in, std::string out, std::string out0, std::string out1, std::string out2,std::string seg);

    bool dynamic         = false;
    bool save            = false;
    bool noseg           = false;
    bool agnostic        = false;
    bool crop            = false;
    bool saveTxt         = false;
    bool saveMask        = false;
    bool rotate          = false;
    int target_size      = 640;
    float prob_threshold = 0.25f;
    float nms_threshold  = 0.45f;
    int max_object       = 100;
private:
    ncnn::Net net;
    std::vector<std::string> class_names;
    int class_count=0;
    double inference_time;
    std::string in_blob;
    std::string out_blob;
    std::string out1_blob;
    std::string out2_blob;
    std::string out3_blob;
    std::string seg_blob;
private:
    /// @brief Draw all the objects at once.
    /// This function is a combination of draw_mask, draw_label and cv::rectangle
    /// @param bgr background image to be draw on. This function will makes a clone of the background image to avoid drawing on the original image
    /// @param objects object vector contain all the detected object in the image
    /// @param colorMode determine the color for each object to be draw ( bounding box and feature mask )
    /// @return image with all the objects draw on
    cv::Mat draw_objects(cv::Mat bgr, const std::vector<Object>& objects, int colorMode = byIndex);

    void draw_mask(cv::Mat& bgr, const cv::Mat& mask, const unsigned char* color);

    void draw_RotatedRect(cv::Mat& bgr, const cv::RotatedRect& rect, const cv::Scalar& cc, int thickness = 1);

    std::vector<cv::Point> mask2segment(const cv::Mat& mask, int strategy = largestContour);

    void draw_label(cv::Mat& bgr, const cv::Rect2f& rect, std::string label);

    cv::Mat applyMask(const cv::Mat& bgr, const cv::Mat& mask);

    cv::Mat getRotatedRectImg(const cv::Mat& src, const cv::RotatedRect& rr);
};