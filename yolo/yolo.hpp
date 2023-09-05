#pragma once
#include <ncnn/net.h>
#include <opencv2/core/mat.hpp>

#include "common.hpp"

#ifdef NDEBUG
#define TIME_LOG(name)
#else
#define TIME_LOG(name) Timer timer(name)
#endif // NDEBUG

#ifdef BENCHMARK
#define LOG(message)
#else
#define LOG(message) std::cout << message
#endif // _DEBUG

class Yolo {
public:
    Yolo();

    ~Yolo();

    int load(const char* bin, const char* param);

    int detect(const cv::Mat& bgr, std::vector<Object>& objects);

    int detect_dynamic(const cv::Mat& bgr, std::vector<Object>& objects);

    void get_blob_name(const char* in, const char* out, const char* out1, const char* out2, const char* out3, const char* seg);

public:
    bool dynamic         = false;
    bool agnostic        = false;
    int target_size      = 640;
    float prob_threshold = 0.25f;
    float nms_threshold  = 0.45f;
    int max_object       = 100;

private:
    ncnn::Net net;

    const char* in_blob   = "in0";
    const char* out_blob  = "out0";
    const char* out1_blob = "out1";
    const char* out2_blob = "out2";
    const char* out3_blob = "out3";
    const char* seg_blob  = "seg";

private:
    void slice(const ncnn::Mat& in, ncnn::Mat& out, int start, int end, int axis);

    void interp(const ncnn::Mat& in, const float& scale, const int& out_w, const int& out_h, ncnn::Mat& out);

    void reshape(const ncnn::Mat& in, ncnn::Mat& out, int c, int h, int w, int d);

    void sigmoid(ncnn::Mat& bottom);

    void matmul(const std::vector<ncnn::Mat>& bottom_blobs, ncnn::Mat& top_blob);

    void decode_mask(const ncnn::Mat& mask_feat,
                     const int& img_w, const int& img_h,
                     const ncnn::Mat& mask_proto,
                     const ncnn::Mat& in_pad,
                     const int& wpad, const int& hpad,
                     ncnn::Mat& mask_pred_result);

    inline float intersection_area(const Object& a, const Object& b);

    void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right);

    void qsort_descent_inplace(std::vector<Object>& faceobjects);

    float sigmoid(float x);

    float relu(float x);

    // For using permute output layer
    void generate_proposals(const ncnn::Mat& anchors, int stride, const ncnn::Mat& in_pad, const ncnn::Mat& feat_blob, float prob_threshold, std::vector<Object>& objects);

    // For using convolution output layer
    void generate_proposals(const ncnn::Mat& anchors, int stride, const ncnn::Mat& feat_blob, float prob_threshold, std::vector<Object>& objects);

    void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold, bool agnostic = true);
};