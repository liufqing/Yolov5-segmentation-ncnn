#pragma once
#include <ncnn/net.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <stdio.h>
#include <string>
#include <float.h>
#include <vector>
#include <fstream>

#include "common.hpp"

class Yolo {
public:
    Yolo();
    ~Yolo();
    int load(const std::string& bin, const std::string& param);

    int detect(const cv::Mat& bgr, std::vector<Object>& objects);

    int detect_dynamic(const cv::Mat& bgr, std::vector<Object>& objects);

    void draw_segment(cv::Mat& bgr, cv::Mat mask, const unsigned char* color);

    void draw_segment(cv::Mat& bgr, const std::vector<Object>& objects, int mode);
    /**
     * @brief 
     * 
     * @param bgr : background image to be draw on
     * @param objects : object vector contain all the detected object in the image
     * @param mode : determine the color for each object to be draw ( bounding box and feature mask ) - default = 1 
     * @note
     * 
     * mode = 0 : color object by class index
     * mode = 1 : color object by object number index
     */
    void draw_objects(cv::Mat& bgr, const std::vector<Object>& objects, int mode = 1);

    void video(cv::VideoCapture capture);

    void image(cv::Mat in, std::string outputPath);

    void get_class_names(std::string data);

    void get_blob_name(std::string in, std::string out, std::string out0, std::string out1, std::string out2,std::string seg);

    bool dynamic         = false;
    bool save            = false;
    bool noseg           = false;
    bool agnostic        = true;
    int target_size      = 640;
    float prob_threshold = 0.25;
    float nms_threshold  = 0.45;
    float mask_conf      = 0.5;
private:
    ncnn::Net net;
    std::vector<std::string> class_names;
    std::vector<Object> objects;
    int class_count=0;
    std::string in_blob;
    std::string out_blob;
    std::string out1_blob;
    std::string out2_blob;
    std::string out3_blob;
    std::string seg_blob;
};