#include "yolo.hpp"
#include "parser.hpp"

int main(int argc, char* argv[]) {
    InputParser argument(argc, argv);

    //set folder
    std::string inputFolder     = "../input";
    std::string outputFolder    = "../output/seg";
    std::string modelFolder     = "../models/seg";
    std::string dataFolder      = "../data";

    //set default argument
    std::string model     =           argument.setDefaultArgument("-model", "yolov5s-seg-idcard-best.ncnn");
    std::string data      =           argument.setDefaultArgument("-data", "idcard.txt");
    std::string input     =           argument.setDefaultArgument("-input", "cmnd5.jpg");
    int size              = std::stoi(argument.setDefaultArgument("-size", "640"));
    float conf            = std::stof(argument.setDefaultArgument("-conf", "0.4"));
    float nms             = std::stof(argument.setDefaultArgument("-nms", "0.45"));
    bool dynamic          =           argument.cmdOptionExists("-dynamic");
    bool save             =           argument.cmdOptionExists("-save");
    bool noseg            =           argument.cmdOptionExists("-noseg");
    bool agnostic         =           argument.cmdOptionExists("-agnostic");

    std::cout << argument.argNum() << " argument(s) passed";

    std::string inputPath  = inputFolder  +   "/" + input;
    std::string outputPath = outputFolder +   "/" + input;
    std::string dataPath   = dataFolder   +   "/" + data;
    std::string bin        = modelFolder  +   "/" + model + ".bin";
    std::string param      = modelFolder  +   "/" + model + ".param";

    std::cout   << "\nmodel     = " << bin
                << "\nparam     = " << param 
                << "\ninput     = " << inputPath
                << "\ndata      = " << dataPath
                << "\nsize      = " << size
                << "\nconf      = " << conf
                << "\nnms       = " << nms
                << "\ndynamic   = " << dynamic
                << "\nsave      = " << save
                << "\nagnostic  = " << agnostic
                << "\nnoseg     = " << noseg 
                << "\n------------------------------------------------" <<std::endl;

    Yolo Yolov5;

    if (Yolov5.load(bin, param)) {
        return -1;
    }
    Yolov5.get_class_names(dataPath);
    Yolov5.dynamic        = dynamic;
    Yolov5.save           = save;
    Yolov5.target_size    = size;
    Yolov5.prob_threshold = conf;
    Yolov5.nms_threshold  = nms;
    Yolov5.noseg          = noseg;
    Yolov5.agnostic       = agnostic;

    if (input == "0") {
        cv::VideoCapture capture;
        capture.open(0);
        Yolov5.video(capture);
        return 0;
    }
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
    return 0;
}
