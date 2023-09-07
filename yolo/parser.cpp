#include "pch.hpp"
#include "parser.hpp"

InputParser::InputParser(int& argc, char** argv){
    for (int i = 1; i < argc; ++i)
        this->tokens.push_back(std::string(argv[i]));
}

const std::string& InputParser::get(const std::string& option){
    std::vector<std::string>::const_iterator itr;
    itr = std::find(this->tokens.begin(), this->tokens.end(), option);
    if (itr != this->tokens.end() && ++itr != this->tokens.end()) {
        return *itr;
    }
    static const std::string empty_string("");
    return empty_string;
}

bool InputParser::has(const std::string& option){
    if (std::find(this->tokens.begin(), this->tokens.end(), option) != this->tokens.end()) {
        argCount++;
        return 1;
    }
    return 0;
}

const std::string& InputParser::get(const std::string& option, const std::string& def){
    if (has(option))
        return get(option);
    else
        return def;
}

const int InputParser::get(const std::string& option, const int& def) {
    if (has(option))
		return std::stoi(get(option));
	else
		return def;
}

const float InputParser::get(const std::string& option, const float& def){
    if (has(option))
		return std::stof(get(option));
	else
		return def;
}

int InputParser::getArgCount() {
    return argCount;
}
