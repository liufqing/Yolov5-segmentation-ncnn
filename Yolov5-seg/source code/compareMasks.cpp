#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <math.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

int main(int argc, char* argv[]) {
	const char* file1 = argv[1];
	const char* file2 = argv[2];

	cv::Mat mask1 = cv::imread(file1, cv::IMREAD_GRAYSCALE);
	cv::Mat mask2 = cv::imread(file2, cv::IMREAD_GRAYSCALE);
	cv::Mat difference;
	cv::subtract(mask1, mask2, difference);

	cv::imshow("difference", difference);
	cv::waitKey(0);

	cv::imwrite("difference.png", difference);
}
