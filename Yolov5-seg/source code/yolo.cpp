#include "yolo.hpp"

Yolo::Yolo() {
	net.opt.use_vulkan_compute = false;
	net.opt.num_threads = 4;
    in_blob   = "in0";
    out_blob  = "out0";
    out1_blob = "out1";
    out2_blob = "out2";
    out3_blob = "out3";
    seg_blob  = "seg";
}

Yolo::~Yolo() {
    net.clear();
}

int Yolo::load(const std::string& bin, const std::string& param) {
    if (net.load_param(param.c_str())){
        return -1;
    }
    if (net.load_model(bin.c_str())){
        return -1;
    }
    return 0;
}

void Yolo::get_class_names(std::string data) {
    std::ifstream file(data);
    std::string name = "";
    int count = 0;
    while (std::getline(file, name)) {
        class_names.push_back(name);
        count++;
    }
    class_count = count;
}
void Yolo::get_blob_name(std::string in, std::string out, std::string out1, std::string out2, std::string out3, std::string seg){
    in_blob = in;
    out_blob = out;
    out1_blob = out1;
    out2_blob = out2;
    out3_blob = out3;
    seg_blob = seg;
}

int Yolo::detect(const cv::Mat& bgr, std::vector<Object>& objects) {
    clock_t tStart = clock();
    // load image, resize and pad to 640x640
    const int img_w = bgr.cols;
    const int img_h = bgr.rows;

    // solve resize scale
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
    const int wpad = target_size - w;
    const int hpad = target_size - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);

    // apply yolov5 pre process, that is to normalize 0~255 to 0~1
    const float norm_vals[3] = { 1 / 255.f, 1 / 255.f, 1 / 255.f };
    in_pad.substract_mean_normalize(0, norm_vals);

    //inference
    ncnn::Extractor ex = net.create_extractor();
    ex.input(in_blob.c_str(), in_pad);
    ncnn::Mat out;
    ex.extract(out_blob.c_str(), out);
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
    ex.extract(seg_blob.c_str(), mask_proto);

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
    
    inference_time = (double)(clock() - tStart) / CLOCKS_PER_SEC ;

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
    for (int i = 0; i < objCount; i++){
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

int Yolo::detect_dynamic(const cv::Mat& bgr, std::vector<Object>& objects) {
    clock_t tStart = clock();

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
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);

    // apply yolov5 pre process, that is to normalize 0~255 to 0~1
    const float norm_vals[3] = { 1 / 255.f, 1 / 255.f, 1 / 255.f };
    in_pad.substract_mean_normalize(0, norm_vals);

    // yolov5 model inference
    ncnn::Extractor ex = net.create_extractor();
    ex.input(in_blob.c_str(), in_pad);

    ncnn::Mat out1;
    ncnn::Mat out2;
    ncnn::Mat out3;
    ex.extract(out1_blob.c_str(), out1);
    ex.extract(out2_blob.c_str(), out2);
    ex.extract(out3_blob.c_str(), out3);
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
    ex.extract(seg_blob.c_str(), mask_proto);

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

    inference_time = (double)(clock() - tStart) / CLOCKS_PER_SEC;

    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(proposals);

    // apply nms with nms_threshold 
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold, agnostic);

    // collect final result after nms
    int count = picked.size();

    ncnn::Mat mask_feat = ncnn::Mat(32, count, sizeof(float));
    for (int i = 0; i < count; i++) {
        std::copy(proposals[picked[i]].mask_feat.begin(), proposals[picked[i]].mask_feat.end(), mask_feat.row(i));
    }

    ncnn::Mat mask_pred_result;
    decode_mask(mask_feat, img_w, img_h, mask_proto, in_pad, wpad, hpad, mask_pred_result);

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

        objects[i].cv_mask = cv::Mat::zeros(img_h, img_w, CV_32FC1);
        cv::Mat mask = cv::Mat(img_h, img_w, CV_32FC1, (float*)mask_pred_result.channel(i));
        mask(objects[i].rect).copyTo(objects[i].cv_mask(objects[i].rect));
    }

    return 0;
}

cv::Mat Yolo::draw_objects(cv::Mat bgr, const std::vector<Object>& objects, int mode) {
    cv::Mat out = bgr.clone();
    int objCount  = objects.size();
    std::cout << "Objects count = " << objCount <<std::endl;

    int color_index = 0;
    for (size_t i = 0; i < objCount; i++) {
        const Object& obj = objects[i];

        char line[256];
        //class-index confident center-x center-y box-width box-height
        sprintf_s(line, "%i %f %i %i %i %i", obj.label, obj.prob, (int)round(obj.rect.tl().x), (int)round(obj.rect.tl().y), (int)round(obj.rect.br().x), (int)round(obj.rect.br().y));

        std::cout << line << std::endl;

        if(mode == 0)
            color_index = obj.label;

        const unsigned char* color = colors[color_index];
        cv::Scalar cc(color[0], color[1], color[2]);

        if(mode == 1)
            color_index = i;

        cv::rectangle(out, obj.rect, cc, 1);

        std::string text = class_names[obj.label] + " " + cv::format("%.2f", obj.prob * 100) + "%";

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > out.cols)
            x = out.cols - label_size.width;

        cv::rectangle(out, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)), cv::Scalar(255, 255, 255), -1);
        cv::putText(out, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

        cv::Mat binMask;
        cv::threshold(obj.cv_mask, binMask, 0.5, 1, cv::ThresholdTypes::THRESH_BINARY); // Mask Binarization

        if (saveMask) {
            std::string maskDir = "../output/seg/masks/" + inputNameWithoutExt + "_" + std::to_string(i) + ".jpg";
            cv::imwrite(maskDir,binMask*255);
        }
        //cv::Mat segment = mask2segment(binMask);

        //cv::polylines(out, segment, true, cc, 2);
                
        draw_segment(out, obj.cv_mask, color);

        //Write labels to file
        if (saveTxt) {
            std::string fileName = outputFolder + "/labels/" + inputNameWithoutExt +".txt";
            std::ofstream labelFile;
            labelFile.open(fileName, std::ios_base::app);
            std::cout << "Labels saved to : " << fileName << std::endl;

            labelFile << line << std::endl;

            labelFile.close();
        }
        if (crop)
            crop_object(bgr, binMask, obj.rect);
    }
    return out;
}

void Yolo::crop_object(cv::Mat& bgr, cv::Mat mask, cv::Rect rect){
    cv::Mat RoI         (bgr,  rect); //Region Of Interest
    cv::Mat mask_cropped(mask, rect);
    cv::imshow("RoI",          RoI);
    cv::imshow("Mask Cropped", mask_cropped);
}

cv::Mat Yolo::mask2segment(cv::Mat &mask, int strategy){
    cv::Mat maskCopy;
    mask.convertTo(maskCopy, CV_8U);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(maskCopy, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::Mat segment;
    if (!contours.empty()) {
        if (!strategy) {
            std::vector<cv::Point> concatenatedPoints;
            for (std::vector<cv::Point> contour : contours) {
                concatenatedPoints.insert(concatenatedPoints.end(), contour.begin(), contour.end());
            }
            segment = cv::Mat(concatenatedPoints).reshape(2);
        }
        else {
            std::vector<cv::Point> largestContour = *std::max_element(contours.begin(), contours.end(),
                [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
                    return a.size() < b.size();
                });
            segment = cv::Mat(largestContour).reshape(2);
        }
    }
    else {
        segment = cv::Mat();
    }

    return segment;
}

void Yolo::draw_segment(cv::Mat& bgr, cv::Mat mask, const unsigned char* color) {
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

void Yolo::image(std::string inputPath) {
    cv::Mat in = cv::imread(inputPath);
    if (dynamic)
        detect_dynamic(in, objects);
    else
        detect(in, objects);

    std::cout << "Inference time = " << inference_time << " (seconds)\n";

    std::string inputName = inputPath.substr(inputPath.find_last_of("/\\") + 1);
    std::string outputName = outputFolder + "/" + inputName;
    inputNameWithoutExt = inputName.substr(0, inputName.find_last_of("."));

    cv::Mat out = draw_objects(in, objects ,0);

    cv::imshow("Detect", out);
    cv::waitKey();

    if (save) {
        cv::imwrite(outputName, out);
        std::cout << "\nOutput saved at " << outputName;
    }
}

void Yolo::video(cv::VideoCapture capture) {
    if (capture.isOpened()) {
        std::cout << "Object Detection Started...." << std::endl;

        cv::Mat frame, out;
        do {
            capture >> frame; //extract frame by frame
            if (dynamic)
                detect_dynamic(frame, objects);
            else
                detect(frame, objects);
            out = frame.clone();
            draw_objects(out, objects, 0);
            cv::imshow("Detect", out);

            char key = (char)cv::pollKey();

            if (key == 27 || key == 'q' || key == 'Q') // Press q or esc to exit from window
                break;
        } while (!frame.empty());
    }
    else {
        std::cout << "Could not Open Camera/Video";
    }
}