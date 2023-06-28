// redirecting cout's output thrrough its stream buffer
#include <iostream>     // std::streambuf, std::cout
#include <fstream>      // std::ofstream

int main() {
	std::streambuf* psbuf, * backup;
	std::ofstream output;
	output.open("test.txt");

	backup = std::cout.rdbuf();     // back up cout's streambuf

	psbuf = output.rdbuf();        // get file's streambuf
	std::cout.rdbuf(psbuf);         // assign streambuf to cout

	std::cout << "This is written to the file";

	std::cout.rdbuf(backup);        // restore cout's original streambuf

	output.close();

	return 0;
}