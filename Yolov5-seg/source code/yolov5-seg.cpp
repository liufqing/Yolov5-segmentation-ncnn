#include <ncnn/net.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <float.h>
#include <stdio.h>
#include <vector>
#include <iostream>
#include <filesystem>
#include <string>
#include <fstream>

#include "classNames.h"
#include "colors.h"

namespace fs = std::filesystem;

#define MAX_STRIDE 64
#define DYNAMIC 1

ncnn::Net yolov5;

struct Object{
    cv::Rect_<float> rect;
    int label{};
    float prob{};
    std::vector<float> mask_feat;
    cv::Mat cv_mask;
};

static void slice(const ncnn::Mat& in, ncnn::Mat& out, int start, int end, int axis){
    ncnn::Option opt;
    opt.num_threads = 4;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = false;

    ncnn::Layer* op = ncnn::create_layer("Crop");

    // set param
    ncnn::ParamDict pd;

    ncnn::Mat axes = ncnn::Mat(1);
    axes.fill(axis);
    ncnn::Mat ends = ncnn::Mat(1);
    ends.fill(end);
    ncnn::Mat starts = ncnn::Mat(1);
    starts.fill(start);
    pd.set(9, starts);  // start
    pd.set(10, ends);   // end
    pd.set(11, axes);   //axes

    op->load_param(pd);

    op->create_pipeline(opt);

    // forward
    op->forward(in, out, opt);

    op->destroy_pipeline(opt);

    delete op;
}
static void interp(const ncnn::Mat& in, const float& scale, const int& out_w, const int& out_h, ncnn::Mat& out){
    ncnn::Option opt;
    opt.num_threads = 4;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = false;

    ncnn::Layer* op = ncnn::create_layer("Interp");

    // set param
    ncnn::ParamDict pd;
    pd.set(0, 2);       // resize_type
    pd.set(1, scale);   // height_scale
    pd.set(2, scale);   // width_scale
    pd.set(3, out_h);   // height
    pd.set(4, out_w);   // width

    op->load_param(pd);

    op->create_pipeline(opt);

    // forward
    op->forward(in, out, opt);

    op->destroy_pipeline(opt);

    delete op;
}
static void reshape(const ncnn::Mat& in, ncnn::Mat& out, int c, int h, int w, int d){
    ncnn::Option opt;
    opt.num_threads = 4;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = false;

    ncnn::Layer* op = ncnn::create_layer("Reshape");

    // set param
    ncnn::ParamDict pd;

    pd.set(0, w);           // start
    pd.set(1, h);           // end
    if (d > 0)
        pd.set(11, d);      //axes
    pd.set(2, c);           //axes
    op->load_param(pd);

    op->create_pipeline(opt);

    // forward
    op->forward(in, out, opt);

    op->destroy_pipeline(opt);

    delete op;
}
static void sigmoid(ncnn::Mat& bottom){
    ncnn::Option opt;
    opt.num_threads = 4;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = false;

    ncnn::Layer* op = ncnn::create_layer("Sigmoid");

    op->create_pipeline(opt);

    // forward

    op->forward_inplace(bottom, opt);
    op->destroy_pipeline(opt);

    delete op;
}
static void matmul(const std::vector<ncnn::Mat>& bottom_blobs, ncnn::Mat& top_blob){
    ncnn::Option opt;
    opt.num_threads = 2;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = false;

    ncnn::Layer* op = ncnn::create_layer("MatMul");

    // set param
    ncnn::ParamDict pd;
    pd.set(0, 0);// axis

    op->load_param(pd);

    op->create_pipeline(opt);
    std::vector<ncnn::Mat> top_blobs(1);
    op->forward(bottom_blobs, top_blobs, opt);
    top_blob = top_blobs[0];

    op->destroy_pipeline(opt);

    delete op;
}

static inline float intersection_area(const Object& a, const Object& b){
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right){
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j){
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j){
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
        #pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects) {
    if (faceobjects.empty())
        return;

    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold) {
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++){
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++){
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++){
            const Object& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

inline float fast_exp(float x) {
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

inline float sigmoid(float x) {
    return 1.0f / (1.0f + fast_exp(-x));
}

static void decode_mask(const ncnn::Mat& mask_feat, const int& img_w, const int& img_h,
                        const ncnn::Mat& mask_proto, const ncnn::Mat& in_pad, const int& wpad, const int& hpad,
                        ncnn::Mat& mask_pred_result){
    ncnn::Mat masks;
    matmul(std::vector<ncnn::Mat>{mask_feat, mask_proto}, masks);
    sigmoid(masks);
    reshape(masks, masks, masks.h, in_pad.h / 4, in_pad.w / 4, 0);
    interp(masks, 4.0, 0, 0, masks);
    slice(masks, mask_pred_result, wpad / 2, in_pad.w - wpad / 2, 2);
    slice(mask_pred_result, mask_pred_result, hpad / 2, in_pad.h - hpad / 2, 1);
    interp(mask_pred_result, 1.0, img_w, img_h, mask_pred_result);
}

static void generate_proposals(const ncnn::Mat& anchors, int stride, const ncnn::Mat& in_pad, const ncnn::Mat& feat_blob, float prob_threshold, std::vector<Object>& objects) {
    const int num_grid = feat_blob.h;

    int num_grid_x;
    int num_grid_y;
    // if (in_pad.w > in_pad.h){
    //     num_grid_x = in_pad.w / stride;
    //     num_grid_y = num_grid / num_grid_x;
    // }
    // else{
    //     num_grid_y = in_pad.h / stride;
    //     num_grid_x = num_grid / num_grid_y;
    // }

    num_grid_x = in_pad.w / stride;
    num_grid_y = in_pad.h / stride;
    
    const int num_anchors = anchors.w / 2;
    const int num_class = feat_blob.w - 5 - 32;// -5

    // enumerate all anchor types
    for (int q = 0; q < num_anchors; q++){
        const float anchor_w = anchors[q * 2];
        const float anchor_h = anchors[q * 2 + 1];
        const ncnn::Mat feat = feat_blob.channel(q);
        for (int i = 0; i < num_grid_y; i++){
            for (int j = 0; j < num_grid_x; j++) {
                const float* featptr = feat.row(i * num_grid_x + j);
                float box_score = featptr[4];
                float box_confidence = sigmoid(box_score);
                if(box_confidence >= prob_threshold) {
                    // find class_index with max class_score
                    int class_index = 0;
                    float class_score = -FLT_MAX;
                    for (int k = 0; k < num_class; k++) {
                        float score = featptr[5 + k];
                        if (score > class_score){
                            class_index = k;
                            class_score = score;
                        }
                    }

                    // combined score = box score * class score
                    // apply sigmoid first to get normed 0~1 value
                    float confidence = sigmoid(box_score) * sigmoid(class_score);

                    // filter candidate boxes with combined score >= prob_threshold
                    if (confidence >= prob_threshold){
                        // yolov5/models/yolo.py Detect forward
                        // y = x[i].sigmoid()
                        // y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                        // y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                        
                        float dx = sigmoid(featptr[0]);
                        float dy = sigmoid(featptr[1]);
                        float dw = sigmoid(featptr[2]);
                        float dh = sigmoid(featptr[3]);

                        float cx = (dx * 2.f - 0.5f + j) * stride;
                        float cy = (dy * 2.f - 0.5f + i) * stride;
                        float bw = pow(dw * 2.f, 2) * anchor_w;
                        float bh = pow(dh * 2.f, 2) * anchor_h;

                        // transform candidate box (center-x,center-y,w,h) to (x0,y0,x1,y1)
                        float x0 = cx - bw * 0.5f;
                        float y0 = cy - bh * 0.5f;
                        float x1 = cx + bw * 0.5f;
                        float y1 = cy + bh * 0.5f;

                        // collect candidates
                        Object obj;
                        obj.rect.x = x0;
                        obj.rect.y = y0;
                        obj.rect.width = x1 - x0;
                        obj.rect.height = y1 - y0;
                        obj.label = class_index;
                        obj.prob = confidence;
                        obj.mask_feat.resize(32); //?
                        std::copy(featptr + 5 + num_class, featptr + 5 + num_class + 32, obj.mask_feat.begin()); //?

                        objects.push_back(obj);
                    }
                }
            }
        }
    }
}

#if DYNAMIC
static int detect_yolov5_seg(const cv::Mat& bgr, std::vector<Object>& objects, const int target_size = 640, const float prob_threshold = 0.25f, const float nms_threshold = 0.45f){
    // load image, resize and letterbox pad to multiple of MAX_STRIDE
    int img_w = bgr.cols;
    int img_h = bgr.rows;

    // letterbox pad to multiple of MAX_STRIDE
    int w = img_w;
    int h = img_h;
    float scale = 1.f;
    if (w > h) {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }

    // construct ncnn::Mat from image pixel data, swap order from bgr to rgb
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, w, h);

    // pad to target_size rectangle
    // yolov5/utils/datasets.py letterbox
    int wpad = (w + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - w;
    int hpad = (h + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);//0.f?

    // apply yolov5 pre process, that is to normalize 0~255 to 0~1
    const float norm_vals[3] = { 1 / 255.f, 1 / 255.f, 1 / 255.f };
    in_pad.substract_mean_normalize(0, norm_vals);

    // yolov5 model inference
    ncnn::Extractor ex = yolov5.create_extractor();
    ex.input("images", in_pad); // images or in0

    ncnn::Mat out0;
    ncnn::Mat out1;
    ncnn::Mat out2;
    ex.extract("output", out0); //output or out0
    ex.extract("385", out1); //yolov5n + yolov5s : 385 ;  yolov5l : 619 ;  yolov5x : 736
    ex.extract("405", out2); //yolov5n + yolov5s : 405 ;  yolov5l : 639 ;  yolov5x : 756

    ncnn::Mat mask_proto;
    ex.extract("seg", mask_proto);

    std::vector<Object> proposals;

    // anchor setting from yolov5/models/yolov5s.yaml

    // stride 8
    {
        ncnn::Mat anchors(6);
        anchors[0] = 10.f;
        anchors[1] = 13.f;
        anchors[2] = 16.f;
        anchors[3] = 30.f;
        anchors[4] = 33.f;
        anchors[5] = 23.f;

        std::vector<Object> objects8;
        generate_proposals(anchors, 8, in_pad, out0, prob_threshold, objects8);

        proposals.insert(proposals.end(), objects8.begin(), objects8.end());
    }

    // stride 16
    {
        ncnn::Mat anchors(6);
        anchors[0] = 30.f;
        anchors[1] = 61.f;
        anchors[2] = 62.f;
        anchors[3] = 45.f;
        anchors[4] = 59.f;
        anchors[5] = 119.f;

        std::vector<Object> objects16;
        generate_proposals(anchors, 16, in_pad, out1, prob_threshold, objects16);

        proposals.insert(proposals.end(), objects16.begin(), objects16.end());
    }

    // stride 32
    {
        ncnn::Mat anchors(6);
        anchors[0] = 116.f;
        anchors[1] = 90.f;
        anchors[2] = 156.f;
        anchors[3] = 198.f;
        anchors[4] = 373.f;
        anchors[5] = 326.f;

        std::vector<Object> objects32;
        generate_proposals(anchors, 32, in_pad, out2, prob_threshold, objects32);

        proposals.insert(proposals.end(), objects32.begin(), objects32.end());
    }

    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(proposals);

    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    // collect final result after nms
    int count = picked.size();

    ncnn::Mat mask_feat = ncnn::Mat(32, count, sizeof(float));
    for (int i = 0; i < count; i++){
        std::copy(proposals[picked[i]].mask_feat.begin(), proposals[picked[i]].mask_feat.end(), mask_feat.row(i));
    }

    ncnn::Mat mask_pred_result;
    decode_mask(mask_feat, img_w, img_h, mask_proto, in_pad, wpad, hpad, mask_pred_result);

    objects.resize(count);
    for (int i = 0; i < count; i++){
        objects[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (objects[i].rect.x - (wpad / 2)) / scale;
        float y0 = (objects[i].rect.y - (hpad / 2)) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width - (wpad / 2)) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height - (hpad / 2)) / scale;

        // clip
        x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;

        objects[i].cv_mask = cv::Mat::zeros(img_h, img_w, CV_32FC1);
        cv::Mat mask = cv::Mat(img_h, img_w, CV_32FC1, (float*)mask_pred_result.channel(i));
        mask(objects[i].rect).copyTo(objects[i].cv_mask(objects[i].rect));
    }

    return 0;
}
#else
static int detect_yolov5_seg(const cv::Mat& bgr, std::vector<Object>& objects, const int target_size = 640, const float prob_threshold = 0.25f, const float nms_threshold = 0.45f){
    // load image, resize and pad to 640x640
    const int img_w = bgr.cols;
    const int img_h = bgr.rows;

    // solve resize scale
    int w = img_w;
    int h = img_h;
    float scale = 1.f;
    if (w > h){
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else{
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }

    // construct ncnn::Mat from image pixel data, swap order from bgr to rgb
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, w, h);

    // pad to target_size rectangle
    const int wpad = target_size - w;
    const int hpad = target_size - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);

    // apply yolov5 pre process, that is to normalize 0~255 to 0~1
    const float norm_vals[3] = { 1 / 255.f, 1 / 255.f, 1 / 255.f };
    in_pad.substract_mean_normalize(0, norm_vals);

    // yolov5 model inference
    ncnn::Extractor ex = yolov5.create_extractor();
    ex.input("in0", in_pad); // images or in0
    ncnn::Mat out;
    ex.extract("out0", out); //output or out0
    ncnn::Mat mask_proto;
    ex.extract("out1", mask_proto); //seg or out1

    // enumerate all boxes
    ncnn::Mat anchors(6);
    anchors[0] = 10.f;
    anchors[1] = 13.f;
    anchors[2] = 16.f;
    anchors[3] = 30.f;
    anchors[4] = 33.f;
    anchors[5] = 23.f;
    std::vector<Object> proposals;
    std::vector<Object> objects8;
    generate_proposals(anchors, 8, in_pad, out, prob_threshold, objects8);

    proposals.insert(proposals.end(), objects8.begin(), objects8.end());

    // sort all candidates by score from highest to lowest
    qsort_descent_inplace(proposals);

    // apply non max suppression
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    // collect final result after nms
    const int count = picked.size();

    ncnn::Mat mask_feat = ncnn::Mat(32, count, sizeof(float));
    for (int i = 0; i < count; i++) {
        float* mask_feat_ptr = mask_feat.row(i);
        std::memcpy(mask_feat_ptr, proposals[picked[i]].mask_feat.data(), sizeof(float) * proposals[picked[i]].mask_feat.size());
    }

    ncnn::Mat mask_pred_result;
    decode_mask(mask_feat, img_w, img_h, mask_proto, in_pad, wpad, hpad, mask_pred_result);

    objects.resize(count);
    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (objects[i].rect.x - (wpad / 2)) / scale;
        float y0 = (objects[i].rect.y - (hpad / 2)) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width - (wpad / 2)) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height - (hpad / 2)) / scale;

        // clip
        x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
        objects[i].cv_mask = cv::Mat::zeros(img_h, img_w, CV_32FC1);
        cv::Mat mask = cv::Mat(img_h, img_w, CV_32FC1, (float*)mask_pred_result.channel(i));
        mask(objects[i].rect).copyTo(objects[i].cv_mask(objects[i].rect));
    }

    return 0;
}
#endif

static void draw_segment(cv::Mat& bgr, cv::Mat mask, const unsigned char* color) {
    for (int y = 0; y < bgr.rows; y++) {
        uchar* image_ptr = bgr.ptr(y);
        const float* mask_ptr = mask.ptr<float>(y);
        for (int x = 0; x < bgr.cols; x++) {
            if (mask_ptr[x] >= 0.5) {
                image_ptr[0] = cv::saturate_cast<uchar>(image_ptr[0] * 0.5 + color[2] * 0.5);
                image_ptr[1] = cv::saturate_cast<uchar>(image_ptr[1] * 0.5 + color[1] * 0.5);
                image_ptr[2] = cv::saturate_cast<uchar>(image_ptr[2] * 0.5 + color[0] * 0.5);
            }
            image_ptr += 3;
        }
    }
}

static void draw_objects(cv::Mat& bgr, const std::vector<Object>& objects){
    int color_index = 0;

    for (size_t i = 0; i < objects.size(); i++){
        const Object& obj = objects[i];
        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f (%s)\n", obj.label, obj.prob, obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height, class_names[obj.label]);

        color_index = obj.label ;
        const unsigned char* color = colors[color_index % 80];
        cv::Scalar cc(color[0], color[1], color[2]);

        draw_segment(bgr, obj.cv_mask, color);

        cv::rectangle(bgr, obj.rect, cc, 1);

        char text[256];
        sprintf_s(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > bgr.cols)
            x = bgr.cols - label_size.width;

        cv::rectangle(bgr, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)), cv::Scalar(255, 255, 255), -1);
        cv::putText(bgr, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }
}

int main(int argc, char* argv[]) {
    std::string input, output, model, inputFolder, modelFolder, outputFolder;

    inputFolder = "../input";
    outputFolder = "../output/seg";
    modelFolder = "../models/seg";

    cv::Mat in, out;
    cv::VideoCapture capture;

    std::vector<Object> objects;

    if (argc < 2) {
        model = "preconvert/yolov5s-seg";
        std::cout << "No argument pass. Using default model " << model;
        std::cout << "\nEnter input : ";
        //std::cin >> input;
        input = "cat.bmp";
    }
    else {
        model = argv[1];
        input = argv[2];
    }

    std::string inputPath = inputFolder + "/" + input;
    std::string outputPath = outputFolder + "/" + input;
    std::string bin = modelFolder + "/" + model + ".bin";
    std::string param = modelFolder + "/" + model + ".param";
    
    // fs::path filePath = input;

    if (yolov5.load_param(param.c_str()))
        exit(-1);
    if (yolov5.load_model(bin.c_str()))
        exit(-1);

    yolov5.opt.use_vulkan_compute = false;
    yolov5.opt.num_threads = 4;

    if (input == "0") {
        std::cout << "Using camera\nUsing " << bin << " and " << param << std::endl;
        capture.open(0);
    }
    else {
        in = cv::imread(inputPath, 1);
        std::cout << "Input = " << inputPath << "\nUsing " << bin << " and " << param << std::endl;
        if (!in.empty()) {
            detect_yolov5_seg(in, objects);
            out = in.clone();

            draw_objects(out, objects);
            cv::imshow("Detect", out);
            cv::waitKey();

            std::cout << "\nOutput saved at " << outputPath;
            cv::imwrite(outputPath, out);

            return 0;
        }
        else {
            capture.open(inputPath);
        }
    }

    if (capture.isOpened()) {
        std::cout << "Object Detection Started...." << std::endl;
        do {
            capture >> in; //extract frame by frame
            detect_yolov5_seg(in, objects);

            out = in.clone();
            //draw_objects_seg(out, objects);
            draw_objects(out, objects);
            cv::imshow("Detect", out);

            char key = (char)cv::pollKey();

            if (key == 27 || key == 'q' || key == 'Q') // Press q or esc to exit from window
                break;
        } while (!in.empty());
    }
    else {
        std::cout << "Could not Open Camera/Video";
    }

    return 0;
}
