#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <math.h>
#include <algorithm>
#include <iomanip>

using std::cout, std::endl, std::setw;

int main(int argc, char* argv[]) {
	const char* inputFile = argv[1];
	std::ifstream input(inputFile);
	float left, right, total = 0;
	float min = 180, max = 0;
	int max_angle = 0, min_angle = 0;
	int count = 0, skip = 0;

	std::vector<std::pair<float, float>> angles;

	while (input >> left >> right){
		angles.push_back(std::make_pair(left, right));
	}

	std::sort(angles.begin(), angles.end(), [] (auto& left, auto& right) {return left.first < right.first;});

	cout << setw(10) << "before" << setw(10) << "after" << setw(10) << "diff" << setw(10) << "false" << endl;

	for (auto& angle : angles) {
		float before = angle.first;
		float after = angle.second;

        float diff = abs(before - after);

		cout << setw(10) << before << setw(10) << after << setw(10) << diff;

        if (diff > 90) {
			cout << setw(10) << "*" << endl;
            skip++;
            continue;
        }
        total += diff;
        count++;
        if (diff > max) {
            max = diff;
            max_angle = before;
        }
        if (diff < min) {
            min = diff;
            min_angle = before;
        }
		cout << endl;
	}

	cout << "------------------------------------------------" << std::endl;
	cout << "Average = " << total / count << endl;
	cout << "Max = " << max << " at " << max_angle << " degree" << endl;
	cout << "Min = " << min << " at " << min_angle << " degree" << endl;
	cout << "False angle = " << skip << endl;

	return 0;
}