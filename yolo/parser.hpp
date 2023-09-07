#pragma once
#include <string>
#include <vector>

struct Argument {
    Argument();

    Argument(const std::string& name, const std::string& value = "");

private:
    std::string name;
    std::string value;
};

class Parser {
public:
    /// <summary>
    /// Parser constructor
    /// </summary>
    /// <param name="argc">number of command line arguments (from main())</param>
    /// <param name="argv">array of command line arguments (from main())</param>
    Parser(int& argc, char** argv);

    /// <summary>
    /// check if argument exist
    /// </summary>
    /// <param name="name">argument name</param>
    /// <returns></returns>
    bool has(const std::string& name);

    /// <summary>
    /// get argument by name, if not exist, return default argument
    /// </summary>
    /// <param name="name">argument name</param>
    /// <param name="def">default argument</param>
    /// <returns></returns>
    const std::string get(const std::string& name, const std::string& def = "");

    /// <summary>
   /// get argument by name, if not exist, return default argument
   /// </summary>
   /// <param name="name">argument name</param>
   /// <param name="def">default argument</param>
   /// <returns></returns>
    const int get(const std::string& name, const int& def);

    /// <summary>
    /// get argument by name, if not exist, return default argument
    /// </summary>
    /// <param name="name">argument name</param>
    /// <param name="def">default argument</param>
    /// <returns></returns>
    const float get(const std::string& name, const float& def);

    /// <summary>
    /// Return the number of arguments has passed to program
    /// </summary>
    /// <returns></returns>
    int getArgCount();

private:
    std::vector <std::string> tokens;

    std::vector <Argument> arguments;

    int argCount = 0;
};
