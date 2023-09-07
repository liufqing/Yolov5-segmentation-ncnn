#include "pch.hpp"
#include "parser.hpp"

Parser::Parser(int& argc, char** argv) {
    for (int i = 1; i < argc; ++i)
        this->tokens.push_back(std::string(argv[i]));
}

const std::string& Parser::get(const std::string& name) {
    auto itr = std::find(this->tokens.begin(), this->tokens.end(), name);
    if (itr != this->tokens.end() && ++itr != this->tokens.end()) {
        return *itr;
    }

    return "";
}

bool Parser::has(const std::string& name) {
    names.push_back(name);
    if (std::find(this->tokens.begin(), this->tokens.end(), name) != this->tokens.end()) {
        argCount++;
        return true;
    }
    return false;
}

const std::string& Parser::get(const std::string& name, const std::string& def) {
    if (has(name))
        return get(name);
    else
        return def;
}

const int Parser::get(const std::string& name, const int& def) {
    if (has(name))
        return std::stoi(get(name));
    else
        return def;
}

const float Parser::get(const std::string& name, const float& def) {
    if (has(name))
        return std::stof(get(name));
    else
        return def;
}

int Parser::getArgCount() {
    return argCount;
}
