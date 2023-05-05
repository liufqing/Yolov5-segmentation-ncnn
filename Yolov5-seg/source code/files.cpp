#include "files.h"

void mkdir(std::string& fileName) {
	auto makeDir = std::filesystem::create_directories(fileName);
	if (makeDir)
		std::cout << "\nCreated " << fileName;
}