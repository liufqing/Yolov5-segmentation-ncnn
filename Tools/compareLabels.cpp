#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <math.h>


int main(int argc, char* argv[]) {
	const char* file1 = argv[1];
	const char* file2 = argv[2];

	std::ifstream label1File(file1);
	std::ifstream label2File(file2);


	std::vector<float> labels1;
	std::vector<float> labels2;

	std::string line;
	while (std::getline(label1File, line, ' ')) {
		float value = std::stof(line);
		//value *= 640;
		labels1.push_back(value);
	}

	while (std::getline(label2File, line, ' ')) {
		float value = std::stof(line);
		//value *= 640;
		labels2.push_back(value);
	}
	std::cout << "labels1.size() = " << labels1.size() << std::endl;
	std::cout << "labels2.size() = " << labels2.size() << std::endl;


	std::vector<float> labels_offset;
	labels_offset.resize(labels1.size());

	for (int i = 0; i < labels_offset.size(); i++) {
		labels_offset[i] = abs(labels1[i] - labels2[i]);
	}

	for (int i = 0; i < labels_offset.size(); i++) {
		std::cout << labels_offset[i] << std::endl;
	}
	return 0;
}