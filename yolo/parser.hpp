#pragma once
#include <string>
#include <vector>
class InputParser {
public:
    InputParser(int& argc, char** argv);
    const std::string& getCmdOption(const std::string& option);
    bool cmdOptionExists(const std::string& option);
    const std::string& setDefaultArgument(const std::string& option, const std::string& def);
    const int setDefaultArgument(const std::string& option, const int& def);
    const float setDefaultArgument(const std::string& option, const float& def);

    int argNum();
private:
    std::vector <std::string> tokens;
    int argCount = 0;
};
