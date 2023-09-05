#include "pch.hpp"
#include "utils.hpp"

const unsigned char colors[81][3] = {
    {56,  0,   255},
    {226, 255, 0},
    {0,   94,  255},
    {0,   37,  255},
    {0,   255, 94},
    {255, 226, 0},
    {0,   18,  255},
    {255, 151, 0},
    {170, 0,   255},
    {0,   255, 56},
    {255, 0,   75},
    {0,   75,  255},
    {0,   255, 169},
    {255, 0,   207},
    {75,  255, 0},
    {207, 0,   255},
    {37,  0,   255},
    {0,   207, 255},
    {94,  0,   255},
    {0,   255, 113},
    {255, 18,  0},
    {255, 0,   56},
    {18,  0,   255},
    {0,   255, 226},
    {170, 255, 0},
    {255, 0,   245},
    {151, 255, 0},
    {132, 255, 0},
    {75,  0,   255},
    {151, 0,   255},
    {0,   151, 255},
    {132, 0,   255},
    {0,   255, 245},
    {255, 132, 0},
    {226, 0,   255},
    {255, 37,  0},
    {207, 255, 0},
    {0,   255, 207},
    {94,  255, 0},
    {0,   226, 255},
    {56,  255, 0},
    {255, 94,  0},
    {255, 113, 0},
    {0,   132, 255},
    {255, 0,   132},
    {255, 170, 0},
    {255, 0,   188},
    {113, 255, 0},
    {245, 0,   255},
    {113, 0,   255},
    {255, 188, 0},
    {0,   113, 255},
    {255, 0,   0},
    {0,   56,  255},
    {255, 0,   113},
    {0,   255, 188},
    {255, 0,   94},
    {255, 0,   18},
    {18,  255, 0},
    {0,   255, 132},
    {0,   188, 255},
    {0,   245, 255},
    {0,   169, 255},
    {37,  255, 0},
    {255, 0,   151},
    {188, 0,   255},
    {0,   255, 37},
    {0,   255, 0},
    {255, 0,   170},
    {255, 0,   37},
    {255, 75,  0},
    {0,   0,   255},
    {255, 207, 0},
    {255, 0,   226},
    {255, 245, 0},
    {188, 255, 0},
    {0,   255, 18},
    {0,   255, 75},
    {0,   255, 151},
    {255, 56,  0},
    {245, 255, 0}
};

std::vector<std::string> IMG_FORMATS {"bmp", "dng", "jpg", "jpeg", "mpo", "png", "tif", "tiff", "webp", "pfm"};
std::vector<std::string> VID_FORMATS {"asf", "avi", "gif", "m4v", "mkv", "mov", "mp4", "mpeg", "mpg", "ts", "wmv"};

Utils::Utils() {
    yolo = new Yolo();
}

Utils::Utils(int argc, char** argv) {
    yolo = new Yolo();
    set_arguments(argc, argv);
}

Utils::~Utils() {
    delete yolo;
}

int Utils::run() {
    if (load(this->model))
        return -1;
    get_class_names(this->data);

    std::filesystem::path outputPath = output;
    std::filesystem::path inputPath = input;

    if (not inputPath.has_extension()) {
        this->show = false;
        folder(inputPath, outputPath);
        return 0;
    }

    if (isImage(inputPath)) {
        image(inputPath, outputPath);
        return 0;
    }
    if (input == "0" or isVideo(inputPath)) {
        video(inputPath.string());
        return 0;
    }

    LOG("input type not supported");
    return 1;
}


void Utils::set_arguments(int argc, char** argv) {
    InputParser parser (argc, argv);

    this->model                    = parser.setDefaultArgument("--model", this->model);
    this->data                     = parser.setDefaultArgument("--data", this->data);
    this->input                    = parser.setDefaultArgument("--source", this->input);
    this->output                   = parser.setDefaultArgument("--output", this->output);
    this->crop                     = parser.cmdOptionExists("--crop");
    this->save                     = parser.cmdOptionExists("--save");
    this->saveTxt                  = parser.cmdOptionExists("--save-txt");
    this->saveMask		           = parser.cmdOptionExists("--save-mask");
    this->rotate			       = parser.cmdOptionExists("--rotate");
    this->show                     = parser.cmdOptionExists("--show");
    yolo->target_size              = parser.setDefaultArgument("--size", 640);
    yolo->prob_threshold           = parser.setDefaultArgument("--conf", 0.25f);
    yolo->nms_threshold            = parser.setDefaultArgument("--nms", 0.45f);
    yolo->max_object               = parser.setDefaultArgument("--max-obj", 1);
    yolo->dynamic                  = parser.cmdOptionExists("--dynamic");
    yolo->agnostic                 = parser.cmdOptionExists("--agnostic");

    LOG("model       = " << this->model);
    LOG("\ndata      = " << this->data);
    LOG("\ninput     = " << this->input);
    LOG("\noutput    = " << this->output);
    LOG("\ncrop      = " << this->crop);
    LOG("\nsave      = " << this->save);
    LOG("\nsaveTxt   = " << this->saveTxt);
    LOG("\nsaveMask  = " << this->saveMask);
    LOG("\nrotate    = " << this->rotate);
    LOG("\nshow      = " << this->show);
    LOG("\nsize      = " << yolo->target_size);
    LOG("\nconf      = " << yolo->prob_threshold);
    LOG("\nnms       = " << yolo->nms_threshold);
    LOG("\nmaxObj    = " << yolo->max_object);
    LOG("\ndynamic   = " << yolo->dynamic);
    LOG("\nagnostic  = " << yolo->agnostic);
    LOG("\n------------------------------------------------" << std::endl);

    LOG(parser.argNum() << " argument(s) passed" << std::endl);
}

int Utils::load(const std::string& _model) {
    std::filesystem::path bin        = _model + ".bin";
    std::filesystem::path param      = _model + ".param";
    load(bin, param);
    return 0;
}

int Utils::load(const std::filesystem::path& bin, const std::filesystem::path& param) {
    return yolo->load(bin.string().c_str(), param.string().c_str());
}

void Utils::draw_objects(cv::Mat& bgr, const std::vector<Object>& objects, int colorMode) {
    size_t objCount = objects.size();
    LOG("Objects count = " << objCount << std::endl);

    int color_index = 0;
    for (size_t i = 0; i < objCount; i++) {
        const Object& obj = objects[i];

        char line[256];
        //class-index confident center-x center-y box-width box-height
        sprintf_s(line, "%i %f %i %i %i %i", obj.label, obj.prob, (int) round(obj.rect.tl().x), (int) round(obj.rect.tl().y), (int) round(obj.rect.br().x), (int) round(obj.rect.br().y));

        LOG(line << std::endl);

        if (colorMode == byClass)
            color_index = obj.label;

        const unsigned char* color = colors[color_index];
        cv::Scalar cc(color[0], color[1], color[2]);

        if (colorMode == byIndex)
            color_index = i;

        cv::rectangle(bgr, obj.rect, cc, 1);

        std::string text = class_names[obj.label] + " " + cv::format("%.2f", obj.prob * 100) + "%";

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

        draw_mask(bgr, obj.cv_mask, color);
    }
}

void Utils::draw_label(cv::Mat& bgr, const cv::Rect2f& rect, std::string label) {
    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

    int x = rect.x;
    int y = rect.y - label_size.height - baseLine;
    if (y < 0)
        y = 0;
    if (x + label_size.width > bgr.cols)
        x = bgr.cols - label_size.width;

    cv::rectangle(bgr, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)), cv::Scalar(255, 255, 255), -1);
    cv::putText(bgr, label, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
}

cv::Mat Utils::applyMask(const cv::Mat& bgr, const cv::Mat& mask) {
    cv::Mat binMask;
    cv::threshold(mask, binMask, 0.5, 255, cv::ThresholdTypes::THRESH_BINARY); // Mask Binarization
    cv::Mat maskCopy;
    binMask.convertTo(maskCopy, CV_8U);
    cv::Mat applyMask;
    bgr.copyTo(applyMask, maskCopy);
    return applyMask;
}

std::vector<cv::Point> Utils::mask2segment(const cv::Mat& mask, int strategy) {
    cv::Mat binMask;
    cv::threshold(mask, binMask, 0.5, 255, cv::ThresholdTypes::THRESH_BINARY); // Mask Binarization
    cv::Mat maskCopy;
    binMask.convertTo(maskCopy, CV_8U);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(maskCopy, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<cv::Point> contour;

    if (!contours.size())
        return contour;

    if (strategy == concatenatedContour) {
        for (std::vector<cv::Point> concatenatedPoints : contours) {
            contour.insert(contour.end(), concatenatedPoints.begin(), concatenatedPoints.end());
        }
    } else {
        contour = *std::max_element(contours.begin(), contours.end(),
                                    [] (const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
                                        return a.size() < b.size();
                                    });
    }

    return contour;
}

void Utils::draw_mask(cv::Mat& bgr, const cv::Mat& mask, const unsigned char* color) {
    cv::Mat binMask;
    cv::threshold(mask, binMask, 0.5, 255, cv::ThresholdTypes::THRESH_BINARY); // Mask Binarization
    for (int y = 0; y < bgr.rows; y++) {
        uchar* image_ptr = bgr.ptr(y);
        const float* mask_ptr = binMask.ptr<float>(y);
        for (int x = 0; x < bgr.cols; x++) {
            if (mask_ptr[x]) {
                image_ptr[0] = cv::saturate_cast<uchar>(image_ptr[0] * 0.5 + color[0] * 0.5);
                image_ptr[1] = cv::saturate_cast<uchar>(image_ptr[1] * 0.5 + color[1] * 0.5);
                image_ptr[2] = cv::saturate_cast<uchar>(image_ptr[2] * 0.5 + color[2] * 0.5);
            }
            image_ptr += 3;
        }
    }
}

void Utils::draw_RotatedRect(cv::Mat& bgr, const cv::RotatedRect& rr, const cv::Scalar& cc, int thickness) {
    cv::Point2f vertices[4];
    rr.points(vertices);
    for (int i = 0; i < 4; i++)
        cv::line(bgr, vertices[i], vertices[(i + 1) % 4], cc, thickness);
}

void Utils::image(const std::filesystem::path& inputPath, const std::filesystem::path& outputFolder) {
    cv::Mat in = cv::imread(inputPath.string());
    std::vector<Object> objects;
    if (yolo->dynamic)
        yolo->detect_dynamic(in, objects);
    else
        yolo->detect(in, objects);

    std::string fileName = inputPath.filename().string();
    std::string stem = inputPath.stem().string();
    std::string outputPath = outputFolder.string() + "\\" + fileName;
    std::string labelsFolder = outputFolder.string() + "\\labels";
    std::string labelsPath = labelsFolder + "\\" + stem + ".txt";
    std::string cropFolder = outputFolder.string() + "\\crop";
    std::string maskFolder = outputFolder.string() + "\\mask";
    std::string rotateFolder = outputFolder.string() + "\\rotate";
    std::string anglePath = rotateFolder + "\\" + "angle.txt";

    const size_t objCount = objects.size();
    LOG("Objects count = " << objCount << std::endl);

    int color_index = 0;
    cv::Mat out = in.clone();
    int colorMode = byClass;
    std::string labels;
    for (int i = 0; i < objCount; i++) {
        const Object& obj = objects[i];
        if (colorMode == byClass)
            color_index = obj.label;
        if (colorMode == byIndex)
            color_index = i;

        const unsigned char* color = colors[color_index % 80];
        cv::Scalar cc(color[0], color[1], color[2]);

        char line[256];
        //class-index confident center-x center-y box-width box-height
        sprintf_s(line, "%i %f %i %i %i %i", obj.label, obj.prob, (int) round(obj.rect.tl().x), (int) round(obj.rect.tl().y), (int) round(obj.rect.br().x), (int) round(obj.rect.br().y));
        labels.append(line);
        if (i != objCount - 1)
            labels.append("\n");

        cv::rectangle(out, obj.rect, cc);
        draw_label(out, obj.rect, class_names[obj.label] + " " + cv::format("%.2f", obj.prob * 100) + "%");

        cv::Mat binMask;
        cv::threshold(obj.cv_mask, binMask, 0.5, 255, cv::ThresholdTypes::THRESH_BINARY); // Mask Binarization

        std::vector<cv::Point> contour = mask2segment(obj.cv_mask);
        if (drawContour)
            cv::polylines(out, contour, true, cc, 1);
        else
            draw_mask(out, obj.cv_mask, color);

        std::string saveFileName = stem + "_" + std::to_string(i) + "_" + class_names[obj.label] + ".jpg";

        float rotAngle = 0;
        if (rotate) {
            cv::Mat rotated;
            if (contour.size() < 3)
                rotated = in(obj.rect);
            else {
                cv::RotatedRect rr = cv::minAreaRect(contour);
                //draw_RotatedRect(out, rr, cv::Scalar(0,0,255));
                rotAngle = -getRotatedRectImg(in, rotated, rr);
            }
            cv::utils::fs::createDirectory(rotateFolder);
            std::string rotatePath = rotateFolder + "\\" + saveFileName;
            if (show)
                cv::imshow("Rotated", rotated);
            if (save)
                cv::imwrite(rotatePath, rotated);

            std::ofstream angle, diff;
            angle.open(anglePath, std::ios::app);
            angle << stem << " " << rotAngle << std::endl;
            angle.close();
        }

        if (crop) {
            cv::utils::fs::createDirectory(cropFolder);
            cv::Mat RoI(in, obj.rect); //Region Of Interest
            std::string cropPath = cropFolder + "\\" + saveFileName;
            cv::imwrite(cropPath, RoI);
        }

        if (saveMask) {
            cv::utils::fs::createDirectory(maskFolder);
            std::string maskPath = maskFolder + "\\" + saveFileName;
            cv::imwrite(maskPath, binMask);
        }
    }

    LOG(labels);

    if (show) {
        cv::imshow("Detect", out);
        cv::waitKey();
    }

    if (save) {
        cv::utils::fs::createDirectory(outputFolder.string());
        cv::imwrite(outputPath, out);
        LOG("\nOutput saved at " << outputPath);
    }

    if (saveTxt) {
        cv::utils::fs::createDirectory(labelsFolder);
        std::ofstream txtFile(labelsPath);
        txtFile << labels;
        txtFile.close();
        LOG("\nLabels saved at " << labelsPath);
    }
}

void Utils::folder(const std::filesystem::path& inputPath, const std::filesystem::path& outputFolder) {
    LOG("Auto running on all images in the input folder");
    int count = 0;
    clock_t tStart = clock();
    for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(inputPath)) {
        std::string path = entry.path().string();
        LOG("\n------------------------------------------------" << std::endl);
        LOG(path << std::endl);
        if (isImage(path)) {
            count++;
            image(entry.path(), outputFolder);
        }
        else {
            LOG("skipping non image file");
        }
    }
    auto total = (double) (clock() - tStart) / CLOCKS_PER_SEC;
    double average = total / count;
    LOG("\n------------------------------------------------" << std::endl);
    LOG(count << " images processed" << std::endl);
    LOG("Total time taken: " << total << " seconds" << std::endl);
    LOG("Average time taken: " << average << " seconds" << std::endl);
}

void Utils::image(const std::filesystem::path& inputPath) {
    cv::Mat in = cv::imread(inputPath.string());
    std::vector<Object> objects;
    if (yolo->dynamic)
        yolo->detect_dynamic(in, objects);
    else
        yolo->detect(in, objects);
}

void Utils::video(std::string inputPath) {
    cv::VideoCapture capture;
    if (inputPath == "0") {
        capture.open(0);
    } else {
        capture.open(inputPath);
    }
    if (capture.isOpened()) {
        LOG("Object Detection Started...." << std::endl);
        LOG("Press q or esc to stop" << std::endl);

        std::vector<Object> objects;

        cv::Mat frame;
        size_t frameIndex = 0;
        do {
            capture >> frame; //extract frame by frame
            if (yolo->dynamic)
                yolo->detect_dynamic(frame, objects);
            else
                yolo->detect(frame, objects);
            draw_objects(frame, objects, 0);
            cv::imshow("Detect", frame);
            if (save) {
                cv::utils::fs::createDirectory("..\\frame");
                std::string saveFileName = "..\\frame\\" + std::to_string(frameIndex) + ".jpg";
                cv::imwrite(saveFileName, frame);
                frameIndex++;
            }

            char key = (char) cv::pollKey();

            if (key == 27 || key == 'q' || key == 'Q') // Press q or esc to exit from window
                break;
        } while (!frame.empty());
    } else {
        LOG("Could not Open Camera/Video");
    }
}

float Utils::getRotatedRectImg(const cv::Mat& src, cv::Mat& dst, const cv::RotatedRect& rr) {
    float angle = rr.angle;
    float width = rr.size.width;
    float height = rr.size.height;

    if (rr.size.width < rr.size.height) {
        std::swap(width, height);
        angle = angle - 90;
    }

    float radianAngle = -angle * CV_PI / 180;
    // angle += M_PI; // you may want rotate it upsidedown
    float sinA = sin(radianAngle), cosA = cos(radianAngle);
    float data[6] =
    {
        cosA, -sinA, width / 2.0f - cosA * rr.center.x + sinA * rr.center.y,
        sinA, cosA, height / 2.0f - cosA * rr.center.y - sinA * rr.center.x
    };
    cv::Mat affineMatrix(2, 3, CV_32FC1, data);

    /*
    Alternate way to get affineMatrix matrix
    cv::Mat affineMatrix(2, 3, CV_32FC1);
    affineMatrix.at<float>(0, 0) = cosA;
    affineMatrix.at<float>(0, 1) = sinA;
    affineMatrix.at<float>(0, 2) = width / 2.0f - cosA * rr.center.x - sinA * rr.center.y;
    affineMatrix.at<float>(1, 0) = -sinA;
    affineMatrix.at<float>(1, 1) = cosA;
    affineMatrix.at<float>(1, 2) = height / 2.0f - cosA * rr.center.y + sinA * rr.center.x;
    */
    //cv::Mat affineMatrix = cv::getRotationMatrix2D(rr.center, rr.angle, 1.0);
    //cv::warpAffine(src, result, affineMatrix, src.size(), cv::INTER_CUBIC);

    cv::warpAffine(src, dst, affineMatrix, cv::Size2f(width, height), cv::INTER_CUBIC);

    return angle;
}

void Utils::matPrint(const ncnn::Mat& m) {
    for (int q = 0; q < m.c; q++) {
        const float* ptr = m.channel(q);
        for (int z = 0; z < m.d; z++) {
            for (int y = 0; y < m.h; y++) {
                for (int x = 0; x < m.w; x++) {
                    printf("%f ", ptr[x]);
                }
                ptr += m.w;
                printf("\n");
            }
            printf("\n");
        }
        printf("------------------------\n");
    }
}

void Utils::matVisualize(const char* title, const ncnn::Mat& m, bool save) {
    std::vector<cv::Mat> normed_feats(m.c);

    for (int i = 0; i < m.c; i++) {
        cv::Mat tmp(m.h, m.w, CV_32FC1, (void*) (const float*) m.channel(i));

        cv::normalize(tmp, normed_feats[i], 0, 255, cv::NORM_MINMAX, CV_8U);

        cv::cvtColor(normed_feats[i], normed_feats[i], cv::COLOR_GRAY2BGR);

        // check NaN
        for (int y = 0; y < m.h; y++) {
            const float* tp = tmp.ptr<float>(y);
            uchar* sp = normed_feats[i].ptr<uchar>(y);
            for (int x = 0; x < m.w; x++) {
                float v = tp[x];
                if (v != v) {
                    sp[0] = 0;
                    sp[1] = 0;
                    sp[2] = 255;
                }
                sp += 3;
            }
        }
        if (!save) {
            cv::imshow(title, normed_feats[i]);
            cv::waitKey();
        }
    }

    if (save) {
        int tw = m.w < 10 ? 32 : m.w < 20 ? 16 : m.w < 40 ? 8 : m.w < 80 ? 4 : m.w < 160 ? 2 : 1;
        int th = (m.c - 1) / tw + 1;

        cv::Mat show_map(m.h * th, m.w * tw, CV_8UC3);
        show_map = cv::Scalar(127);

        // tile
        for (int i = 0; i < m.c; i++) {
            int ty = i / tw;
            int tx = i % tw;

            normed_feats[i].copyTo(show_map(cv::Rect(tx * m.w, ty * m.h, m.w, m.h)));
        }
        cv::resize(show_map, show_map, cv::Size(0, 0), 2, 2, cv::INTER_NEAREST);
        //cv::imshow(title, show_map);
        //cv::waitKey();
        cv::imwrite("masks.jpg", show_map);
    }
}

void Utils::get_class_names(const std::string& dataFile) {
    std::ifstream file(dataFile);
    std::string name = "";
    while (std::getline(file, name)) {
        class_names.push_back(name);
    }
}

void Utils::get_class_names(const std::filesystem::path& data) {
    get_class_names(data.string());
}

bool Utils::isImage(const std::string& path) {
    std::string ext = path.substr(path.find_last_of(".") + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), [] (unsigned char c) { return std::tolower(c); });
    return std::find(IMG_FORMATS.begin(), IMG_FORMATS.end(), ext) != IMG_FORMATS.end();
}

bool Utils::isImage(const std::filesystem::path& path) {
    std::string ext = path.extension().string().substr(1);
    std::transform(ext.begin(), ext.end(), ext.begin(), [] (unsigned char c) { return std::tolower(c); });
    return std::find(IMG_FORMATS.begin(), IMG_FORMATS.end(), ext) != IMG_FORMATS.end();
}

bool Utils::isVideo(const std::string& path) {
    std::string ext = path.substr(path.find_last_of(".") + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), [] (unsigned char c) { return std::tolower(c); });
    return std::find(VID_FORMATS.begin(), VID_FORMATS.end(), ext) != VID_FORMATS.end();
}

bool Utils::isVideo(const std::filesystem::path& path) {
    std::string ext = path.extension().string().substr(1);
    std::transform(ext.begin(), ext.end(), ext.begin(), [] (unsigned char c) { return std::tolower(c); });
    return std::find(VID_FORMATS.begin(), VID_FORMATS.end(), ext) != VID_FORMATS.end();
}