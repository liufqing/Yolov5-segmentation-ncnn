#pragma once

#include <ncnn/net.h>
#include <string>
#include <opencv2/core/core.hpp>

#include "common.hpp"

enum strategy {
    concatenatedContour = 0,    //concatenate all segments
    largestContour = 1     //select largest segment
};

enum colorMode {
    byClass = 0,                //color object by class index
    byIndex = 1                 //color object by object number index 
};

class Yolo {
public:
    Yolo();
    ~Yolo();

public:
    int load(const std::string& bin, const std::string& param);

    int load(const std::filesystem::path& bin, const std::filesystem::path& param);

    int detect(const cv::Mat& bgr, std::vector<Object>& objects);

    int detect_dynamic(const cv::Mat& bgr, std::vector<Object>& objects);

    void video(std::string inputPath);

    void image(const std::filesystem::path& inputPath, const std::filesystem::path& outputFolder, bool continuous = false);

    void get_class_names(const std::string& data);

    void get_class_names(const std::filesystem::path& data);

    void get_blob_name(std::string in, std::string out, std::string out0, std::string out1, std::string out2, std::string seg);

    bool dynamic = false;
    bool save = false;
    bool drawContour = false;
    bool agnostic = false;
    bool crop = false;
    bool saveTxt = false;
    bool saveMask = false;
    bool rotate = false;
    int offset = 0;
    int target_size = 640;
    float prob_threshold = 0.25f;
    float nms_threshold = 0.45f;
    int max_object = 100;

private:
    ncnn::Net net;
    std::vector<std::string> class_names;
    int class_count = 0;
    std::string in_blob;
    std::string out_blob;
    std::string out1_blob;
    std::string out2_blob;
    std::string out3_blob;
    std::string seg_blob;
private:
    /// @brief Draw all the objects at once.
    /// This function is a combination of draw_mask, draw_label and cv::rectangle
    /// @param bgr background image to be draw on.
    /// @param objects object vector contain all the detected object in the image
    /// @param colorMode determine the color for each object to be draw ( bounding box and feature mask )
    void draw_objects(cv::Mat& bgr, const std::vector<Object>& objects, int colorMode = byIndex);

    /// @brief draw color mask on the background image.
    /// @param bgr background image to be draw on.
    /// @param mask gray scale mask
    /// @param color color to be draw on the mask
    void draw_mask(cv::Mat& bgr, const cv::Mat& mask, const unsigned char* color);

    void draw_RotatedRect(cv::Mat& bgr, const cv::RotatedRect& rect, const cv::Scalar& cc, int thickness = 1);

    std::vector<cv::Point> mask2segment(const cv::Mat& mask, int strategy = largestContour);

    void draw_label(cv::Mat& bgr, const cv::Rect2f& rect, std::string label);

    cv::Mat applyMask(const cv::Mat& bgr, const cv::Mat& mask);

    float getRotatedRectImg(const cv::Mat& src, cv::Mat& dst, const cv::RotatedRect& rr);
};