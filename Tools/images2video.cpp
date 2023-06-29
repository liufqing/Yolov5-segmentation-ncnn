#include <iostream> // for standard I/O
#include <string>   // for strings
#include <vector>
#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat)
#include <opencv2/highgui/highgui.hpp>  // Video write

int main(int argc, char* argv[]) {
    std::string imagesFolder = argv[1];
	size_t frameCount = 1059;
	cv::VideoWriter outputVideo;
	cv::Mat firstFrame = cv::imread(imagesFolder + "/out_0.jpg");
	outputVideo.open("output.avi",-1, 25, firstFrame.size(), true);
	if (!outputVideo.isOpened()) {
		std::cout << "Could not open the output video for write: " << std::endl;
		return -1;
	}
    for (size_t frameIndex = 0; frameIndex < frameCount; frameIndex++) {
		std::string imagePath = imagesFolder + "/out_" + std::to_string(frameIndex) + ".jpg";
		cv::Mat frame = cv::imread(imagePath);
		outputVideo << frame;
	}

    return 0;
}