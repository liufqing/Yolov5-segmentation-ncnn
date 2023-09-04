#include "yolo.hpp"

#define MAX_STRIDE  64
#define PERMUTE     0 // Using the permute layer output
#define FAST_EXP    1 // Using fast exponential function


Yolo::Yolo(){
}

Yolo::~Yolo() {
    net.clear();
}

int Yolo::load(const char* bin, const char* param) {
    if (net.load_param(param)) {
        return -1;
    }
    if (net.load_model(bin)) {
        return -1;
    }
    return 0;
}

int Yolo::load(const std::filesystem::path& bin, const std::filesystem::path& param) {
    return load(bin.string().c_str(), param.string().c_str());
}

void Yolo::get_blob_name(const char* in, const char* out, const char* out1, const char* out2, const char* out3, const char* seg) {
    in_blob   = in;
    out_blob  = out;
    out1_blob = out1;
    out2_blob = out2;
    out3_blob = out3;
    seg_blob  = seg;
}

int Yolo::detect(const cv::Mat& bgr, std::vector<Object>& objects) {
    Timer timer("Inference");
    // load image, resize and pad to 640x640
    const int img_w = bgr.cols;
    const int img_h = bgr.rows;

    // solve resize scale
    int w = img_w;
    int h = img_h;
    float scale = 1.f;
    if (w > h) {
        scale = (float) target_size / w;
        w = target_size;
        h = h * scale;
    } else {
        scale = (float) target_size / h;
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
    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in_pad.substract_mean_normalize(0, norm_vals);

    //inference
    ncnn::Extractor ex = net.create_extractor();
    ex.input(in_blob, in_pad);
    ncnn::Mat out;
    ex.extract(out_blob, out);
    /*
    The out blob would be a 2-dim tensor with w=85 h=25200

           |cx|cy|bw|bh|box score(1)| per-class scores(80) |
           +--+--+--+--+------------+----------------------+
           |53|50|70|80|    0.11    |0.1 0.0 0.0 0.5 ......|
      all /|  |  |  |  |      .     |           .          |
     boxes |46|40|38|44|    0.95    |0.0 0.9 0.0 0.0 ......|
    (25200)|  |  |  |  |      .     |           .          |
          \|  |  |  |  |      .     |           .          |
           +--+--+--+--+------------+----------------------+

    The out blob would be a 2-dim tensor with w=117 h=25200 (for segment model)

           |cx|cy|bw|bh|box score(1)| per-class scores(80) |mask feature(32)|
           +--+--+--+--+------------+----------------------+----------------+
           |53|50|70|80|    0.11    |0.1 0.0 0.0 0.5 ......|                |
      all /|  |  |  |  |      .     |           .          |                |
     boxes |46|40|38|44|    0.95    |0.0 0.9 0.0 0.0 ......|                |
    (25200)|  |  |  |  |      .     |           .          |                |
          \|  |  |  |  |      .     |           .          |                |
           +--+--+--+--+------------+----------------------+----------------|
    */

    ncnn::Mat mask_proto;
    ex.extract(seg_blob, mask_proto);

    std::vector<Object> proposals;

    const int num_grid = out.h;
    const int num_class = out.w - 5 - 32;
    for (int i = 0; i < num_grid; i++) {
        const float box_score = out.row(i)[4];

        // find class index with max class score
        int class_index = 0;
        float class_score = -FLT_MAX;
        for (int k = 0; k < num_class; k++) {
            float score = out.row(i)[5 + k];
            if (score > class_score) {
                class_index = k;
                class_score = score;
            }
        }

        // combined score = box score * class score
        float score = box_score * class_score;

        // filter candidate boxes with combined score >= prob_threshold
        if (score >= prob_threshold) {
            const float cx = out.row(i)[0]; //center x coordinate
            const float cy = out.row(i)[1]; //center y coordinate
            const float bw = out.row(i)[2]; //box width
            const float bh = out.row(i)[3]; //box height

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
            obj.prob = score;
            obj.mask_feat.resize(32);
            std::copy(out.row(i) + 5 + num_class, out.row(i) + 5 + num_class + 32, obj.mask_feat.begin());

            proposals.push_back(obj);
        }
    }

    // sort all candidates by score from highest to lowest
    qsort_descent_inplace(proposals);

    // apply non max suppression
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold, agnostic);

    // collect final result after nms
    const int count = picked.size();

    ncnn::Mat mask_feat = ncnn::Mat(32, count, sizeof(float));
    for (int i = 0; i < count; i++) {
        std::copy(proposals[picked[i]].mask_feat.begin(), proposals[picked[i]].mask_feat.end(), mask_feat.row(i));
    }

    ncnn::Mat mask_pred_result;
    decode_mask(mask_feat, img_w, img_h, mask_proto, in_pad, wpad, hpad, mask_pred_result);

    int objCount = (count > max_object) ? max_object : count;
    objects.resize(objCount);
    for (int i = 0; i < objCount; i++) {
        objects[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (objects[i].rect.x - (wpad / 2.0)) / scale;
        float y0 = (objects[i].rect.y - (hpad / 2.0)) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width - (wpad / 2.0)) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height - (hpad / 2.0)) / scale;

        // clip
        x0 = std::max(std::min(x0, (float) (img_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float) (img_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float) (img_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float) (img_h - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;

        objects[i].cv_mask = cv::Mat::zeros(img_h, img_w, CV_32FC1);
        cv::Mat mask = cv::Mat(img_h, img_w, CV_32FC1, (float*) mask_pred_result.channel(i));
        mask(objects[i].rect).copyTo(objects[i].cv_mask(objects[i].rect));
    }

    return 0;
}

int Yolo::detect_dynamic(const cv::Mat& bgr, std::vector<Object>& objects) {
    Timer timer("Inference");
    // load image, resize and letterbox pad to multiple of MAX_STRIDE
    int img_w = bgr.cols;
    int img_h = bgr.rows;

    // letterbox pad to multiple of MAX_STRIDE
    int w = img_w;
    int h = img_h;
    float scale = 1.f;
    if (w > h) {
        scale = (float) target_size / w;
        w = target_size;
        h = h * scale;
    } else {
        scale = (float) target_size / h;
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
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);

    // apply yolov5 pre process, that is to normalize 0~255 to 0~1
    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in_pad.substract_mean_normalize(0, norm_vals);

    // yolov5 model inference
    ncnn::Extractor ex = net.create_extractor();
    ex.input(in_blob, in_pad);

    ncnn::Mat out1;
    ncnn::Mat out2;
    ncnn::Mat out3;
    ex.extract(out1_blob, out1);
    ex.extract(out2_blob, out2);
    ex.extract(out3_blob, out3);
    /*
    The out blob would be a 3-dim tensor with w=dynamic h=dynamic c=255=85*3
    We view it as [grid_w,grid_h,85,3] for 3 anchor ratio types

                |<--   dynamic anchor grids     -->|
                |   larger image yields more grids |
                +-------------------------- // ----+
               /| center-x                         |
              / | center-y                         |
             /  | box-w                            |
     anchor-0   | box-h                            |
      +-----+   | box score(1)                     |
      |     |   +----------------                  |
      |     |   | per-class scores(80)             |
      +-----+\  |   .                              |
              \ |   .                              |
               \|   .                              |
                +-------------------------- // ----+
               /| center-x                         |
              / | center-y                         |
             /  | box-w                            |
     anchor-1   | box-h                            |
      +-----+   | box score(1)                     |
      |     |   +----------------                  |
      +-----+   | per-class scores(80)             |
             \  |   .                              |
              \ |   .                              |
               \|   .                              |
                +-------------------------- // ----+
               /| center-x                         |
              / | center-y                         |
             /  | box-w                            |
     anchor-2   | box-h                            |
      +--+      | box score(1)                     |
      |  |      +----------------                  |
      |  |      | per-class scores(80)             |
      +--+   \  |   .                              |
              \ |   .                              |
               \|   .                              |
                +-------------------------- // ----+

    The out blob would be a 3-dim tensor with w=dynamic h=dynamic c=(80 + 5 + 32)*3 = 351
    We view it as [grid_w,grid_h,117,3] for 3 anchor ratio types

                |<--   dynamic anchor grids     -->|
                |   larger image yields more grids |
                +-------------------------- // ----+
               /| center-x                         |
              / | center-y                         |
             /  | box-w                            |
     anchor-0   | box-h                            |
      +-----+   | box score(1)                     |
      |     |   +----------------                  |
      |     |   | per-class scores(80)             |
      +-----+\  |   .                              |
              \ |   .                              |
               \|   .                              |
                | mask_feat(32)                    |
                |   .                              |
                |   .                              |
                |   .                              |
                +-------------------------- // ----+
               /| center-x                         |
              / | center-y                         |
             /  | box-w                            |
     anchor-1   | box-h                            |
      +-----+   | box score(1)                     |
      |     |   +----------------                  |
      +-----+   | per-class scores(80)             |
             \  |   .                              |
              \ |   .                              |
               \|   .                              |
                | mask_feat(32)                    |
                |   .                              |
                |   .                              |
                |   .                              |
                +-------------------------- // ----+
               /| center-x                         |
              / | center-y                         |
             /  | box-w                            |
     anchor-2   | box-h                            |
      +--+      | box score(1)                     |
      |  |      +----------------                  |
      |  |      | per-class scores(80)             |
      +--+   \  |   .                              |
              \ |   .                              |
               \|   .                              |
                | mask_feat(32)                    |
                |   .                              |
                |   .                              |
                |   .                              |
                +-------------------------- // ----+
    */

    ncnn::Mat mask_proto;
    ex.extract(seg_blob, mask_proto);

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

        std::vector<Object> objects;
#if PERMUTE
        generate_proposals(anchors, 8, in_pad, out1, prob_threshold, objects);
#else
        generate_proposals(anchors, 8, out1, prob_threshold, objects);
#endif // PERMUTE

        proposals.insert(proposals.end(), objects.begin(), objects.end());
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

        std::vector<Object> objects;
#if PERMUTE
        generate_proposals(anchors, 16, in_pad, out2, prob_threshold, objects);
#else
        generate_proposals(anchors, 16, out2, prob_threshold, objects);
#endif // PERMUTE

        proposals.insert(proposals.end(), objects.begin(), objects.end());
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

        std::vector<Object> objects;
#if PERMUTE
        generate_proposals(anchors, 32, in_pad, out3, prob_threshold, objects);
#else
        generate_proposals(anchors, 32, out3, prob_threshold, objects);
#endif // PERMUTE

        proposals.insert(proposals.end(), objects.begin(), objects.end());
    }

    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(proposals);

    // apply nms with nms_threshold 
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold, agnostic);

    // collect final result after nms
    const int count = picked.size();

    ncnn::Mat mask_feat = ncnn::Mat(32, count, sizeof(float));
    for (int i = 0; i < count; i++) {
        std::copy(proposals[picked[i]].mask_feat.begin(), proposals[picked[i]].mask_feat.end(), mask_feat.row(i));
    }

    ncnn::Mat mask_pred_result;
    decode_mask(mask_feat, img_w, img_h, mask_proto, in_pad, wpad, hpad, mask_pred_result);

    int objCount = (count > max_object) ? max_object : count;
    objects.resize(objCount);
    for (int i = 0; i < objCount; i++) {
        objects[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (objects[i].rect.x - (wpad / 2.0)) / scale;
        float y0 = (objects[i].rect.y - (hpad / 2.0)) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width - (wpad / 2.0)) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height - (hpad / 2.0)) / scale;

        // clip
        x0 = std::max(std::min(x0, (float) (img_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float) (img_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float) (img_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float) (img_h - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;

        objects[i].cv_mask = cv::Mat::zeros(img_h, img_w, CV_32FC1);
        cv::Mat mask = cv::Mat(img_h, img_w, CV_32FC1, (float*) mask_pred_result.channel(i));
        mask(objects[i].rect).copyTo(objects[i].cv_mask(objects[i].rect));
    }

    return 0;
}

void Yolo::slice(const ncnn::Mat& in, ncnn::Mat& out, int start, int end, int axis) {
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
void Yolo::interp(const ncnn::Mat& in, const float& scale, const int& out_w, const int& out_h, ncnn::Mat& out) {
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
void Yolo::reshape(const ncnn::Mat& in, ncnn::Mat& out, int c, int h, int w, int d) {
    ncnn::Option opt;
    opt.num_threads = 4;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = false;

    ncnn::Layer* op = ncnn::create_layer("Reshape");

    // set param
    ncnn::ParamDict pd;

    pd.set(0, w);           //start
    pd.set(1, h);           //end
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
void Yolo::sigmoid(ncnn::Mat& bottom) {
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
void Yolo::matmul(const std::vector<ncnn::Mat>& bottom_blobs, ncnn::Mat& top_blob) {
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

void Yolo::decode_mask(const ncnn::Mat& mask_feat, const int& img_w, const int& img_h,
                       const ncnn::Mat& mask_proto, const ncnn::Mat& in_pad, const int& wpad, const int& hpad,
                       ncnn::Mat& mask_pred_result) {
    ncnn::Mat masks;
    matmul(std::vector<ncnn::Mat>{mask_feat, mask_proto}, masks);
    sigmoid(masks);
    reshape(masks, masks, masks.h, in_pad.h / 4, in_pad.w / 4, 0);
    interp(masks, 4.0, 0, 0, masks);
    slice(masks, mask_pred_result, wpad / 2, in_pad.w - wpad / 2, 2);
    slice(mask_pred_result, mask_pred_result, hpad / 2, in_pad.h - hpad / 2, 1);
    interp(mask_pred_result, 1.0, img_w, img_h, mask_pred_result);
}

inline float Yolo::intersection_area(const Object& a, const Object& b) {
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

void Yolo::qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right) {
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j) {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j) {
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

void Yolo::qsort_descent_inplace(std::vector<Object>& faceobjects) {
    if (faceobjects.empty())
        return;

    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

void Yolo::nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold, bool agnostic) {
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++) {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++) {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int) picked.size(); j++) {
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

#if FAST_EXP
inline float fast_exp(float x) {
    union {
        uint32_t i;
        float f;
    } v {};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

inline float Yolo::sigmoid(float x) {
    return 1.0f / (1.0f + fast_exp(-x));
}
#else
inline float Yolo::sigmoid(float x) {
    return static_cast<float>(1.f / (1.f + exp(-x)));
}
#endif // FAST_EXP

inline float Yolo::relu(float x) {
    if (x > 0)
        return x;
    else
        return 0;
}

void Yolo::generate_proposals(const ncnn::Mat& anchors, int stride, const ncnn::Mat& in_pad, const ncnn::Mat& feat_blob, float prob_threshold, std::vector<Object>& objects) {
    const int num_grid = feat_blob.h;
    int num_grid_x;
    int num_grid_y;
    if (in_pad.w > in_pad.h) {
        num_grid_x = in_pad.w / stride;
        num_grid_y = num_grid / num_grid_x;
    } else {
        num_grid_y = in_pad.h / stride;
        num_grid_x = num_grid / num_grid_y;
    }
    const int num_anchors = anchors.w / 2;
    const int num_class = feat_blob.w - 5 - 32;
    // enumerate all anchor types
    for (int q = 0; q < num_anchors; q++) {
        const float anchor_w = anchors[q * 2];
        const float anchor_h = anchors[q * 2 + 1];
        for (int i = 0; i < num_grid_y; i++) {
            for (int j = 0; j < num_grid_x; j++) {
                float box_score = feat_blob.channel(q).row(i * num_grid_x + j)[4];
                float box_confidence = sigmoid(box_score);
                if (box_confidence >= prob_threshold) {
                    // find class_index with max class_score
                    int class_index = 0;
                    float class_score = -FLT_MAX;
                    for (int k = 0; k < num_class; k++) {
                        float score = feat_blob.channel(q).row(i * num_grid_x + j)[5 + k];
                        if (score > class_score) {
                            class_index = k;
                            class_score = score;
                        }
                    }

                    // combined score = box score * class score
                    float score = sigmoid(box_score) * sigmoid(class_score); // apply sigmoid first to get normed 0~1 value

                    // filter candidate boxes with combined score >= prob_threshold
                    if (score >= prob_threshold) {
                        // yolov5/models/yolo.py Detect forward
                        // y = x[i].sigmoid()
                        // y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                        // y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                        float dx = sigmoid(feat_blob.channel(q).row(i * num_grid_x + j)[0]);
                        float dy = sigmoid(feat_blob.channel(q).row(i * num_grid_x + j)[1]);
                        float dw = sigmoid(feat_blob.channel(q).row(i * num_grid_x + j)[2]);
                        float dh = sigmoid(feat_blob.channel(q).row(i * num_grid_x + j)[3]);

                        float cx = (dx * 2.f - 0.5f + j) * stride;  //center x coordinate
                        float cy = (dy * 2.f - 0.5f + i) * stride;  //cennter y coordinate
                        float bw = pow(dw * 2.f, 2.f) * anchor_w;     //box width
                        float bh = pow(dh * 2.f, 2.f) * anchor_h;     //box height

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
                        obj.prob = score;
                        obj.mask_feat.resize(32);
                        std::copy(feat_blob.channel(q).row(i * num_grid_x + j) + 5 + num_class, feat_blob.channel(q).row(i * num_grid_x + j) + 5 + num_class + 32, obj.mask_feat.begin());
                        objects.push_back(obj);
                    }
                }
            }
        }
    }
}

void Yolo::generate_proposals(const ncnn::Mat& anchors, int stride, const ncnn::Mat& feat_blob, float prob_threshold, std::vector<Object>& objects) {
    const int num_grid_x = feat_blob.w;
    const int num_grid_y = feat_blob.h;

    const int num_anchors = anchors.w / 2;
    const int num_class = feat_blob.c / num_anchors - 5 - 32;

    const int feat_offset = num_class + 5 + 32;
    // enumerate all anchor types
    for (int q = 0; q < num_anchors; q++) {
        const float anchor_w = anchors[q * 2];
        const float anchor_h = anchors[q * 2 + 1];
        for (int i = 0; i < num_grid_y; i++) {
            for (int j = 0; j < num_grid_x; j++) {
                float box_score = feat_blob.channel(q * feat_offset + 4).row(i)[j];
                float box_confidence = sigmoid(box_score);
                if (box_confidence >= prob_threshold) {
                    // find class_index with max class_score
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
                    float score = sigmoid(box_score) * sigmoid(class_score); // apply sigmoid first to get normed 0~1 value

                    // filter candidate boxes with combined score >= prob_threshold
                    if (score >= prob_threshold) {
                        // yolov5/models/yolo.py Detect forward
                        // y = x[i].sigmoid()
                        // y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                        // y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                        float dx = sigmoid(feat_blob.channel(q * feat_offset + 0).row(i)[j]);
                        float dy = sigmoid(feat_blob.channel(q * feat_offset + 1).row(i)[j]);
                        float dw = sigmoid(feat_blob.channel(q * feat_offset + 2).row(i)[j]);
                        float dh = sigmoid(feat_blob.channel(q * feat_offset + 3).row(i)[j]);

                        float cx = (dx * 2.f - 0.5f + j) * stride;  //center x coordinate
                        float cy = (dy * 2.f - 0.5f + i) * stride;  //cennter y coordinate
                        float bw = pow(dw * 2.f, 2.f) * anchor_w;     //box width
                        float bh = pow(dh * 2.f, 2.f) * anchor_h;     //box height

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
                        obj.prob = score;
                        for (int c = 0; c < 32; c++)
                            obj.mask_feat.push_back((float) feat_blob.channel(q * feat_offset + 5 + num_class + c).row(i)[j]);
                        objects.push_back(obj);
                    }
                }
            }
        }
    }
}