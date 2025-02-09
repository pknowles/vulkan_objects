// Copyright (c) 2025 Pyarelal Knowles, MIT License

#include <iostream>
#include <filesystem>
#include <pugixml.hpp>
#include <inja/inja.hpp>
#include <regex>

namespace fs = std::filesystem;

inja::json toJson(const pugi::xml_node& node) {
    inja::json result = inja::json::object();
    result["name"] = node.name();
    std::ostringstream oss;
    for (pugi::xml_node child : node)
        child.print(oss);
    result["inner"] = oss.str();
    inja::json attributes = inja::json::object();
    for(const auto& a : node.attributes())
        attributes[a.name()] = a.value();
    result["attributes"] = std::move(attributes);
    return result;
};

int main(int argc, char** argv) {
    if(argc != 4){
        std::cout << "Usage: ./generate <vk.xml> <template.hpp.txt> <output.hpp>\n";
        return EXIT_FAILURE;
    }

    fs::path specFilename(argv[1]);
    fs::path templateFilename(argv[2]);
    fs::path outputFilename(argv[3]);

    std::ofstream outputFile(outputFilename);
    if(!outputFile.good()){
        std::cout << "Failed to open output file '" << outputFilename << "'\n";
        return EXIT_FAILURE;
    }

    pugi::xml_document spec;
    {
        pugi::xml_parse_result result = spec.load_file(specFilename.string().c_str());
        if (!result) {
            std::cerr << "Failed to parse XML: " << result.description() << std::endl;
            return EXIT_FAILURE;
        }
    }

    inja::Environment env;

    env.add_callback("find", 1, [&spec](inja::Arguments& args) {
        pugi::xpath_node path = spec.select_node(args.at(0)->get<std::string>().c_str());
        return toJson(path.node());
    });

    env.add_callback("findall", 1, [&spec](inja::Arguments& args) {
        inja::json result = inja::json::array();
        for(const pugi::xpath_node& path : spec.select_nodes(args.at(0)->get<std::string>().c_str()))
            result.push_back(toJson(path.node()));
        return result;
    });

    env.add_callback("search", 2, [](inja::Arguments& args){
        auto pattern = args.at(0)->get<std::string>();
        auto text = args.at(1)->get<std::string>();
        std::regex re(pattern);
        std::smatch match;
        return std::regex_search(text, match, re) ? match[0].str() : "";
    });

    env.add_callback("sub", 3, [](inja::Arguments& args) {
        auto pattern = args.at(0)->get<std::string>();
        auto replace = args.at(1)->get<std::string>();
        auto text = args.at(2)->get<std::string>();
        std::regex re(pattern);
        return inja::json(std::regex_replace(text, re, replace));
    });

    env.add_callback("slice", 3, [](inja::Arguments& args) {
        auto s = args.at(0)->get<std::string>();
        auto a = args.at(1)->get<ptrdiff_t>();
        auto b = args.at(2)->get<ptrdiff_t>();
        if (a < 0)
            a += s.size();
        if (b < 0)
            b += s.size();
        a = std::min(s.size(), size_t(a));
        b = std::min(s.size(), size_t(b));
        return inja::json(s.substr(a, b - a));
    });

    inja::Template templatArghReservedKeyword;
    try {
        templatArghReservedKeyword = env.parse_template(templateFilename.string());
    } catch ( const std::exception& e ){
        std::cout << "Error loading template file '" << outputFilename << "':\n" << e.what() << "\n";
        return EXIT_FAILURE;
    }

    inja::json data;
    try {
        env.render_to(outputFile, templatArghReservedKeyword, data);
    } catch (const inja::RenderError& e) {
        std::cout << "Error rendering template '" << outputFilename << "':\n" << e.what() << "\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
