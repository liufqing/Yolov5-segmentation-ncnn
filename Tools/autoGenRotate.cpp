#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <opencv2/core/utils/filesystem.hpp>

using std::cout;
using std::endl;

int main(int argc, char* argv[]){
	const char* input = argv[1];
	const char* output = argv[2];
	cv::Mat src = cv::imread(input);
	std::filesystem::path inputPath = input;
	cv::Mat dst;
	cv::Size size(src.cols,src.cols);
	cv::Point2f center(src.cols / 2, src.rows / 2);
	std::string outputFolder = "autogen/" + inputPath.stem().string();
	cv::utils::fs::createDirectory(outputFolder);
	for (int i = 0; i < 360; i++) {
		cv::Mat affineMatrix = cv::getRotationMatrix2D(center, i, 0.8);
		cv::warpAffine(src, dst, affineMatrix, src.size(), cv::INTER_NEAREST, cv::BORDER_REPLICATE);
		cv::imwrite(outputFolder + "/" + std::to_string(i) + ".jpg", dst);
		cout << "Generated " << i+1 << " image" << endl;
	}

	//cv::imshow("dst", dst)
	//cv::waitKey();

	return 0;
}