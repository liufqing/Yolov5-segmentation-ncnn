#include "pch.hpp"
#include "parser.hpp"

Parser::Parser(int& argc, char** argv) {
    for (int i = 1; i < argc; ++i)
        this->tokens.push_back(std::string(argv[i]));
}

bool Parser::has(const std::string& name) {
    auto itr = std::find(this->tokens.begin(), this->tokens.end(), name);
    if (itr != this->tokens.end()) {
        arguments.push_back(Argument(name, "true"));
        argCount++;
        return true;
    }

    arguments.push_back(Argument(name, "false"));
    return false;
}

const std::string Parser::get(const std::string& name, const std::string& def) {
    std::string ret = def;

    auto itr = std::find(this->tokens.begin(), this->tokens.end(), name);
    if (itr != this->tokens.end() && ++itr != this->tokens.end()) {
        argCount++;
        ret = *itr;
    }

    arguments.push_back(Argument(name, ret));
    return ret;
}

const int Parser::get(const std::string& name, const int& def) {
    return std::stoi(get(name,std::to_string(def)));
}

const float Parser::get(const std::string& name, const float& def) {
    return std::stof(get(name, std::to_string(def)));
}

int Parser::getArgCount() {
    return argCount;
}

Argument::Argument() {
}

Argument::Argument(const std::string& name, const std::string& value) {
    this->name = name;
    this->value = value;
}