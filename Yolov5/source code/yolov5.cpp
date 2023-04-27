#include "yolo.hpp"
#include "parser.hpp"

int main(int argc, char* argv[]) {
    InputParser argument(argc, argv);

    //set folder
    std::string inputFolder = "../input";
    std::string outputFolder = "../output";
    std::string modelFolder = "../models";
    std::string dataFolder = "../data";

    //set default argument
    std::string model       = argument.setDefaultArgument("-model", "yolov5s.ncnn");
    std::string data        = argument.setDefaultArgument("-data", "coco128.txt");
    std::string input       = argument.setDefaultArgument("-input", "cat.bmp");
    std::string in_blob     = argument.setDefaultArgument("-in", "in0");
    std::string out_blob    = argument.setDefaultArgument("-out", "out0");
    std::string out0_blob   = argument.setDefaultArgument("-out0", "193");
    std::string out1_blob   = argument.setDefaultArgument("-out1", "207");
    std::string out2_blob   = argument.setDefaultArgument("-out2", "222");
    bool dynamic            = argument.cmdOptionExists("-dynamic");
    bool save               = argument.cmdOptionExists("-save");
    int size      = std::stoi(argument.setDefaultArgument("-size","640"));
    float conf    = std::stof(argument.setDefaultArgument("-conf", "0.25"));
    float nms     = std::stof(argument.setDefaultArgument("-nms", "0.45"));

    std::string inputPath = inputFolder + "/" + input;
    std::string outputPath = outputFolder + "/" + input;
    std::string dataPath = dataFolder + "/" + data;
    std::string bin = modelFolder + "/" + model + ".bin";
    std::string param = modelFolder + "/" + model + ".param";

    std::cout   << "input = " << inputPath
                << "\nmodel =  " << bin
                << "\nparam = " << param 
                << "\ndata = " << dataPath
                << "\nsize = " << size
                << "\nconf = " << conf
                << "\nnms = " << nms
                << "\ndynamic = " << dynamic 
                << "\nsave = " << save << std::endl;

    Yolo Yolov5;

    Yolov5.load(bin,param);
    Yolov5.get_class_names(dataPath);
    Yolov5.dynamic = dynamic;
    Yolov5.save = save;
    Yolov5.target_size = size;
    Yolov5.prob_threshold = conf;
    Yolov5.nms_threshold = nms;
    Yolov5.get_blob_name(in_blob,out_blob,out0_blob,out1_blob,out2_blob);

    if (input == "0") {
        cv::VideoCapture capture;
        capture.open(0);
        Yolov5.video(capture);
        return 0;
    }
    else {
        cv::Mat in = cv::imread(inputPath, 1);
        if (!in.empty()) {
            Yolov5.image(in, outputPath);
            return 0;
        }
        else {
            cv::VideoCapture capture;
            capture.open(inputPath);
            Yolov5.video(capture);
        }
    }
    return 0;
}