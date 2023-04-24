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

#include "classNames.hpp"
#include "colors.hpp"

namespace fs = std::filesystem;

#define MAX_STRIDE 32

ncnn::Net yolo;

struct Object{
    cv::Rect_<float> rect;
    int label{};
    float prob{};
    std::vector<float> mask_feat;
    cv::Mat cv_mask;
};

struct GridAndStride{
    int grid0;
    int grid1;
    int stride;
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
    pd.set(9, starts);  //start
    pd.set(10, ends);   //end
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

    pd.set(0, w);// start
    pd.set(1, h);// end
    if (d > 0)
        pd.set(11, d);//axes
    pd.set(2, c);//axes
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

static void softmax(ncnn::Mat& bbox_pred){
    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_packing_layout = false;

    ncnn::Layer* softmax = ncnn::create_layer("Softmax");

    // set param
    ncnn::ParamDict pd;
    pd.set(0, 1); // axis
    pd.set(1, 1);
    softmax->load_param(pd);


    softmax->create_pipeline(opt);

    softmax->forward_inplace(bbox_pred, opt);

    softmax->destroy_pipeline(opt);

    delete softmax;
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

static void qsort_descent_inplace(std::vector<Object>& faceobjects){
    if (faceobjects.empty())
        return;

    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold){
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++){
        areas[i] = faceobjects[i].rect.width * faceobjects[i].rect.height;
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
    // slice(masks, mask_pred_result, (wpad / 2) / 4, (in_pad.w - wpad / 2) / 4, 2);
    // slice(mask_pred_result, mask_pred_result, (hpad / 2) / 4, (in_pad.h - hpad / 2) / 4, 1);
    // interp(mask_pred_result, 4.0, img_w, img_h, mask_pred_result);
}

static void generate_grids_and_stride(const int target_w, const int target_h, std::vector<int>& strides, std::vector<GridAndStride>& grid_strides){
    for (int i = 0; i < (int)strides.size(); i++){
        int stride = strides[i];
        int num_grid_w = target_w / stride;
        int num_grid_h = target_h / stride;
        for (int g1 = 0; g1 < num_grid_h; g1++){
            for (int g0 = 0; g0 < num_grid_w; g0++){
                GridAndStride gs;
                gs.grid0 = g0;
                gs.grid1 = g1;
                gs.stride = stride;
                grid_strides.push_back(gs);
            }
        }
    }
}

static void generate_proposals(std::vector<GridAndStride> grid_strides, const ncnn::Mat& feat_blob, float prob_threshold, std::vector<Object>& objects){
    const int num_points = grid_strides.size();
    const int num_class = 80;
    const int reg_max_1 = 16;

    for (int i = 0; i < num_points; i++){
        const float* scores = feat_blob.row(i) + 4 * reg_max_1;

        // find class_index with max class_score
        int class_index = -1;
        float class_score = -FLT_MAX;
        for (int k = 0; k < num_class; k++){
            float score = scores[k];
            if (score > class_score){
                class_index = k;
                class_score = score;
            }
        }
        float confidence = sigmoid(class_score);
        if (confidence >= prob_threshold){
            ncnn::Mat bbox_pred(reg_max_1, 4, (void*)feat_blob.row(i));
            softmax(bbox_pred);

            float pred_ltrb[4];
            for (int k = 0; k < 4; k++){
                float dis = 0.f;
                const float* dis_after_sm = bbox_pred.row(k);
                for (int l = 0; l < reg_max_1; l++){
                    dis += l * dis_after_sm[l];
                }

                pred_ltrb[k] = dis * grid_strides[i].stride;
            }

            float pb_cx = (grid_strides[i].grid0 + 0.5f) * grid_strides[i].stride;
            float pb_cy = (grid_strides[i].grid1 + 0.5f) * grid_strides[i].stride;

            float x0 = pb_cx - pred_ltrb[0];
            float y0 = pb_cy - pred_ltrb[1];
            float x1 = pb_cx + pred_ltrb[2];
            float y1 = pb_cy + pred_ltrb[3];

            Object obj;
            obj.rect.x = x0;
            obj.rect.y = y0;
            obj.rect.width = x1 - x0;
            obj.rect.height = y1 - y0;
            obj.label = class_index;
            obj.prob = confidence;
            obj.mask_feat.resize(32);
            std::copy(feat_blob.row(i) + 64 + num_class, feat_blob.row(i) + 64 + num_class + 32, obj.mask_feat.begin());
            objects.push_back(obj);
        }
    }
}

static int detect_yolov8(const cv::Mat& bgr, std::vector<Object>& objects, const int target_size = 640, const float prob_threshold = 0.25f, const float nms_threshold = 0.45f){
    // load image, resize and letterbox pad to multiple of MAX_STRIDE
    int img_w = bgr.cols;
    int img_h = bgr.rows;

    // pad to multiple of MAX_STRIDE
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
    int wpad = (w + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - w;
    int hpad = (h + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 0.f);

    const float norm_vals[3] = { 1 / 255.f, 1 / 255.f, 1 / 255.f };
    in_pad.substract_mean_normalize(0, norm_vals);

    // yolov8 model inference
    ncnn::Extractor ex = yolo.create_extractor();
    ex.input("images", in_pad);

    ncnn::Mat out;
    ex.extract("output", out);

    ncnn::Mat mask_proto;
    ex.extract("seg", mask_proto);
    
    std::vector<Object> proposals;
 
    std::vector<int> strides = { 8, 16, 32 };
    std::vector<GridAndStride> grid_strides;
    generate_grids_and_stride(in_pad.w, in_pad.h, strides, grid_strides);

    std::vector<Object> objects8;
    generate_proposals(grid_strides, out, prob_threshold, objects8);
    
    proposals.insert(proposals.end(), objects8.begin(), objects8.end());

    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(proposals);

    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

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
        // color_index++;
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
        model = "preconvert/yolov8s-seg";
        std::cout << "No argument pass. Using default model\n";
        std::cout << "Enter input : ";
        std::cin >> input;
    }
    else if (argc == 2) {
        model = "preconvert/yolov8s-seg";
        std::cout << "Using default model\n";
        input = argv[1];
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

    if (yolo.load_param(param.c_str()))
        exit(-1);
    if (yolo.load_model(bin.c_str()))
        exit(-1);

    yolo.opt.use_vulkan_compute = false;
    yolo.opt.num_threads = 4;

    if (input == "0") {
        std::cout << "Using camera\nUsing " << bin << " and " << param << std::endl;
        capture.open(0);
    }
    else {
        in = cv::imread(inputPath, 1);
        std::cout << "Input = " << inputPath << "\nUsing " << bin << " and " << param << std::endl;
        if (!in.empty()) {
            detect_yolov8(in, objects);
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
            detect_yolov8(in, objects);

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
