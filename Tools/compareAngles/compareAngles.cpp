#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <math.h>

using std::cout, std::endl;


int main(int argc, char* argv[]) {
	const char* inputFile = argv[1];
	std::ifstream angle(inputFile);
	float left, right, total = 0;
	float min = 180, max = 0;
	int count = 0, skip = 0;
	while (angle >> left >> right){
		float diff = abs(left - right);
		cout << left << " " << right << " " << diff << endl;
		if (diff > 90) {
			skip++;
			continue;
		}
		total += diff;
		count++;
		if (diff > max) max = diff;
		if (diff < min) min = diff;
	}
	cout << "------------------------------------------------" << std::endl;
	cout << "Average = " << total / count << endl;
	cout << "False angle = " << skip << endl;
	cout << "Max = " << max << endl;
	cout << "Min = " << min << endl;

	return 0;
}