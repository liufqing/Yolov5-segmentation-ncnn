#include "yolo.hpp"
#include "parser.hpp"

int main(int argc, char* argv[]) {
    InputParser argument(argc, argv);

    //set folder path
    std::string inputFolder     = "../input";
    std::string outputFolder    = "../output/seg";
    std::string modelFolder     = "../models/seg";
    std::string dataFolder      = "../data";

    //set default argument
    std::string model     = argument.setDefaultArgument("--model", "yolov5s-seg-idcard-2.ncnn");
    std::string data      = argument.setDefaultArgument("--data", "idcard.txt");
    std::string input     = argument.setDefaultArgument("--source", "7.jpg");
    std::string output	  = argument.setDefaultArgument("--output", outputFolder);
    int size              = argument.setDefaultArgument("--size", 640);
    float conf            = argument.setDefaultArgument("--conf", 0.25f);
    float nms             = argument.setDefaultArgument("--nms", 0.45f);
    int maxObj            = argument.setDefaultArgument("--max-obj", 100);
    bool dynamic          = argument.cmdOptionExists("--dynamic");
    bool noseg            = argument.cmdOptionExists("--noseg");
    bool agnostic         = argument.cmdOptionExists("--agnostic");
    bool crop             = argument.cmdOptionExists("--crop");
    bool save             = argument.cmdOptionExists("--save");
    bool saveTxt          = argument.cmdOptionExists("--save-txt");
    bool saveMask		  = argument.cmdOptionExists("--save-mask");

    std::cout << argument.argNum() << " argument(s) passed";

    std::string inputPath  = inputFolder  +   "/" + input;
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
                << "\nmaxObj    = " << maxObj
                << "\ndynamic   = " << dynamic
                << "\nnoseg     = " << noseg
                << "\nagnostic  = " << agnostic
                << "\ncrop      = " << crop
                << "\nsave      = " << save
                << "\nsaveTxt   = " << saveTxt
                << "\nsaveMask  = " << saveMask
                << "\n------------------------------------------------" <<std::endl;

    Yolo Yolov5;

    // load param and bin, assuming param and bin file has the same name
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
    Yolov5.max_object     = maxObj;
    Yolov5.saveTxt		  = saveTxt;
    Yolov5.crop			  = crop;
    Yolov5.saveMask		  = saveMask;
    Yolov5.outputFolder   = outputFolder;

    if (input == "all") {
        std::cout << "Input is empty, running on default input folder" ;
        for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(inputFolder)) {
			std::string path = entry.path().string();
            std::cout << "\n------------------------------------------------" << std::endl;
            std::cout << path << std::endl;
            if (isImage(path)) {
			    Yolov5.image(path);
			}
		}
		return EXIT_SUCCESS;
    }

    if (input == "cam") {
        cv::VideoCapture capture;
        capture.open(0);
        Yolov5.video(capture);
        return EXIT_SUCCESS;
    }
    if (isImage(inputPath)) {
        Yolov5.image(inputPath);
        return EXIT_SUCCESS;
    }
    if (isVideo(inputPath)) {
        cv::VideoCapture capture;
        capture.open(inputPath);
        Yolov5.video(capture);
        return EXIT_SUCCESS;
    }

    std::cout << "input type not supported";
    return EXIT_FAILURE;
}
