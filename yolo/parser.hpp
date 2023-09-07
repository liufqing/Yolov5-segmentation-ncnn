#pragma once
#include <string>
#include <vector>
class Parser {
public:
    Parser(int& argc, char** argv);

    bool has(const std::string& option);

    const std::string& get(const std::string& option);

    const std::string& get(const std::string& option, const std::string& def);

    const int get(const std::string& option, const int& def);

    const float get(const std::string& option, const float& def);

    /// <summary>
    /// Return the number of arguments has passed to program
    /// </summary>
    /// <returns></returns>
    int getArgCount();

private:
    std::vector <std::string> tokens;
    int argCount = 0;
};
