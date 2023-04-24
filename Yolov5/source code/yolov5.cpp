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
};
static inline float intersection_area(const Object& a, const Object& b){
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right){
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
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

static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold, bool agnostic = false) {
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++) {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++) {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++) {
            const Object& b = faceobjects[picked[j]];

            if (!agnostic && a.label != b.label)
                continue;

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

inline float fast_exp(float x){
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}
inline float sigmoid(float x){
    return 1.0f / (1.0f + fast_exp(-x));
}

// static inline float sigmoid(float x){
//     return static_cast<float>(1.f / (1.f + exp(-x)));
// }

static void generate_proposals(const ncnn::Mat& anchors, int stride, const ncnn::Mat& in_pad, const ncnn::Mat& feat_blob, float prob_threshold, std::vector<Object>& objects) {
    // const int num_grid_x = feat_blob.w;
    // const int num_grid_y = feat_blob.h;

    int num_grid_x;
    int num_grid_y;
    num_grid_x = in_pad.w / stride;
    num_grid_y = in_pad.h / stride;

    const int num_anchors = anchors.w / 2;
    const int num_class = feat_blob.c / num_anchors - 5;

    const int feat_offset = num_class + 5;

    // enumerate all anchor types
    for (int q = 0; q < num_anchors; q++) {
        const float anchor_w = anchors[q * 2];
        const float anchor_h = anchors[q * 2 + 1];
        for (int i = 0; i < num_grid_y; i++) {
            for (int j = 0; j < num_grid_x; j++) {
                float box_score = feat_blob.channel(q * feat_offset + 4).row(i)[j];
                float box_confidence = sigmoid(box_score);
                if(box_confidence >= prob_threshold) {
                    // find class index with max class score
                    int class_index = 0;
                    float class_score = -FLT_MAX;
                    for (int k = 0; k < num_class; k++) {
                        float score = feat_blob.channel(q * feat_offset + 5 + k).row(i)[j];
                        if (score > class_score) {
                            class_index = k;
                            class_score = score;
                        }
                    }

                    // combined score = box score * class score
                    // apply sigmoid first to get normed 0~1 value
                    float confidence = sigmoid(box_score) * sigmoid(class_score);

                    // filter candidate boxes with combined score >= prob_threshold
                    if (confidence >= prob_threshold) {
                        // yolov5/models/yolo.py Detect forward
                        // y = x[i].sigmoid()
                        // y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                        // y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh

                        float dx = sigmoid(feat_blob.channel(q * feat_offset + 0).row(i)[j]);
                        float dy = sigmoid(feat_blob.channel(q * feat_offset + 1).row(i)[j]);
                        float dw = sigmoid(feat_blob.channel(q * feat_offset + 2).row(i)[j]);
                        float dh = sigmoid(feat_blob.channel(q * feat_offset + 3).row(i)[j]);

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

                        objects.push_back(obj);
                    }
                }
            }
        }
    }
}  

#if DYNAMIC
static int detect_yolov5(const cv::Mat& bgr, std::vector<Object>& objects,  const int target_size = 640, const float prob_threshold = 0.25f, const float nms_threshold = 0.45f){
    // load image, resize and letterbox pad to multiple of MAX_STRIDE
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
    // yolov5/utils/datasets.py letterbox
    const int wpad = (w + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - w;
    const int hpad = (h + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);

    // apply yolov5 pre process, that is to normalize 0~255 to 0~1
    const float norm_vals[3] = { 1 / 255.f, 1 / 255.f, 1 / 255.f };
    in_pad.substract_mean_normalize(0, norm_vals);

    // yolov5 model inference
    ncnn::Extractor ex = yolov5.create_extractor();
    ex.input("in0", in_pad); //images or in0

    ncnn::Mat out0;
    ncnn::Mat out1;
    ncnn::Mat out2;
    // extract intermediate blob before anchor box generation instead of the very last "out0"
    // the three blob names can be easily discovered in netron visualizer
    // they are the outputs of the last three convolution, before any memory layout transformation
    // each out blob is the feat for a grid scale stride(8, 16, 32)
    ex.extract("193", out0); //193 or 258 or 323 or 388
    ex.extract("207", out1); //207 or 272 or 337 or 402
    ex.extract("222", out2); //222 or 287 or 352 or 417

    // the out blob would be a 3-dim tensor with w=dynamic h=dynamic c=255=85*3
    // we view it as [grid_w,grid_h,85,3] for 3 anchor ratio types

    //
    //            |<--   dynamic anchor grids     -->|
    //            |   larger image yields more grids |
    //            +-------------------------- // ----+
    //           /| center-x                         |
    //          / | center-y                         |
    //         /  | box-w                            |
    // anchor-0   | box-h                            |
    //  +-----+   | box score(1)                     |
    //  |     |   +----------------                  |
    //  |     |   | per-class scores(80)             |
    //  +-----+\  |   .                              |
    //          \ |   .                              |
    //           \|   .                              |
    //            +-------------------------- // ----+
    //           /| center-x                         |
    //          / | center-y                         |
    //         /  | box-w                            |
    // anchor-1   | box-h                            |
    //  +-----+   | box score(1)                     |
    //  |     |   +----------------                  |
    //  +-----+   | per-class scores(80)             |
    //         \  |   .                              |
    //          \ |   .                              |
    //           \|   .                              |
    //            +-------------------------- // ----+
    //           /| center-x                         |
    //          / | center-y                         |
    //         /  | box-w                            |
    // anchor-2   | box-h                            |
    //  +--+      | box score(1)                     |
    //  |  |      +----------------                  |
    //  |  |      | per-class scores(80)             |
    //  +--+   \  |   .                              |
    //          \ |   .                              |
    //           \|   .                              |
    //            +-------------------------- // ----+
    //

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

    // sort all candidates by score from highest to lowest
    qsort_descent_inplace(proposals);

    // apply non max suppression
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    // collect final result after nms
    const int count = picked.size();

    objects.resize(count);
    for (int i = 0; i < count; i++) {
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
    }

    return 0;
}
#else
static int detect_yolov5(const cv::Mat& bgr, std::vector<Object>& objects, const int target_size = 640, const float prob_threshold = 0.25f, const float nms_threshold = 0.45f){
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

    // the out blob would be a 2-dim tensor with w=85 h=25200
    //
    //        |cx|cy|bw|bh|box score(1)| per-class scores(80) |
    //        +--+--+--+--+------------+----------------------+
    //        |53|50|70|80|    0.11    |0.1 0.0 0.0 0.5 ......|
    //   all /|  |  |  |  |      .     |           .          |
    //  boxes |46|40|38|44|    0.95    |0.0 0.9 0.0 0.0 ......|
    // (25200)|  |  |  |  |      .     |           .          |
    //       \|  |  |  |  |      .     |           .          |
    //        +--+--+--+--+------------+----------------------+
    //

    // enumerate all boxes
    std::vector<Object> proposals;
    for (int i = 0; i < out.h; i++) {
        const float* ptr = out.row(i);

        const int num_class = 80;

        const float box_score = ptr[4];

        // find class index with the biggest class score among all classes
        int class_index = 0;
        float class_score = -FLT_MAX;
        for (int k = 0; k < num_class; k++) {
            float score = ptr[5 + k];
            if (score > class_score) {
                class_index = k;
                class_score = score;
            }
        }

        // combined score = box score * class score
        float confidence = box_score * class_score;

        // filter candidate boxes with combined score >= prob_threshold
        if (confidence >= prob_threshold) {
            const float cx = ptr[0]; //center-x
            const float cy = ptr[1]; //center-y
            const float bw = ptr[2]; //box-width
            const float bh = ptr[3]; //box-height

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

            proposals.push_back(obj);
        }
    }

    // sort all candidates by score from highest to lowest
    qsort_descent_inplace(proposals);

    // apply non max suppression
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    // collect final result after nms
    const int count = picked.size();
    
    objects.resize(count);
    for (int i = 0; i < count; i++) {
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
    }

    return 0;
}
#endif
static void draw_objects(const cv::Mat& brg, const std::vector<Object>& objects){
    int color_index = 0;

    for (size_t i = 0; i < objects.size(); i++){
        const Object& obj = objects[i];
        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f (%s)\n", obj.label, obj.prob, obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height, class_names[obj.label]);
        
        color_index = obj.label;
        const unsigned char* color = colors[color_index % 80];
        // color_index++;
        cv::Scalar cc(color[0], color[1], color[2]);

        cv::rectangle(brg, obj.rect, cc, 1);

        char text[256];
        sprintf_s(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > brg.cols)
            x = brg.cols - label_size.width;

        cv::rectangle(brg, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)), cv::Scalar(255, 255, 255), -1);
        cv::putText(brg, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }
}

int main(int argc, char* argv[]) {
    std::string input, output, model, inputFolder, modelFolder, outputFolder;
    
    inputFolder = "../input";
    outputFolder = "../output";
    modelFolder = "../models";

    cv::Mat in,out;
    cv::VideoCapture capture;

    std::vector<Object> objects;

    if (argc < 2) {
        model = "pnnx/yolov5s.ncnn";
        std::cout << "No argument pass. Using default model " << model;
        std::cout << "\nEnter input : ";
        //std::cin >> input;
        input = "cat.jpg";
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
            detect_yolov5(in, objects);
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
            detect_yolov5(in, objects);

            out = in.clone();
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