// Copyright (c) 2025 Pyarelal Knowles, MIT License

#include <filesystem>
#include <inja/inja.hpp>
#include <iostream>
#include <pugixml.hpp>
#include <regex>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace fs = std::filesystem;

struct Command {
    std::string    name;
    pugi::xml_node node;
    std::smatch    match;
    std::string    object;
    std::string    owner;
    bool           plural;
};

std::string inner(const pugi::xml_node& node) {
    std::ostringstream oss;
    for (pugi::xml_node child : node)
        child.print(oss);
    return oss.str();
}

inja::json toJson(const pugi::xml_node& node) {
    inja::json result = inja::json::object();
    result["name"] = node.name();
    result["inner"] = inner(node);
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

    std::filesystem::create_directories(outputFilename.parent_path());
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
    env.set_trim_blocks(true);
    env.set_lstrip_blocks(true);

    env.add_callback("find", 1, [&spec](inja::Arguments& args) {
        pugi::xpath_node path = spec.select_node(args.at(0)->get<std::string>().c_str());
        inja::json result;
        if(path && path.node())
            result = inja::json(path.node().value());
        else if (path)
            result = inja::json(path.attribute().value());
        return result;
    });

    env.add_callback("findall", 1, [&spec](inja::Arguments& args) {
        inja::json result = inja::json::array();
        for(const pugi::xpath_node& path : spec.select_nodes(args.at(0)->get<std::string>().c_str()))
        {
            if(path.node())
                result.push_back(path.node().value());
            else
                result.push_back(path.attribute().value());
        }
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

    inja::json handles = inja::json::array();
    {
        std::regex createFunc("vk(Create|Allocate)(.*)([A-Z]{2,})?");
        std::regex destroyFunc("vk(Destroy|Free)(.*)([A-Z]{2,})?");
        std::regex typeBaseStrip("^Vk|[A-Z]{2,}$");
        std::regex suffixPattern("[A-Z]{2,}$");
        std::regex pluralStrip("(.*)(ies|es|s)$");

        std::unordered_map<std::string, Command> destroyCommands;
        for(const pugi::xpath_node& command : spec.select_nodes("//commands/command[@api='vulkan' or not(@api)]"))
        {
            pugi::xml_node node = command.node();
            std::string destroyFuncName = node.select_node("proto/name/text()").node().value();
            std::smatch destroyMatch;
            if(std::regex_match(destroyFuncName, destroyMatch, destroyFunc))
            {
                std::string object;
                pugi::xpath_node countNode = node.select_node("param/name[contains(text(),'Count')]");
                bool plural = static_cast<bool>(countNode);
                if(plural)
                    object = node.select_node("param[last()]/type/text()").node().value();
                else
                    object = node.select_node("param[position() = (last() - 1)]/type/text()").node().value();
                if(destroyCommands.count(object))
                    throw std::runtime_error("Duplicate destroy functions for type " + object +
                                             ": " + destroyFuncName + " and " +
                                             destroyCommands[object].name + "\n");
                std::string owner = node.select_node("param[1]/type/text()").node().value();
                if(owner == object)
                    owner = "void";
                destroyCommands[object] = {std::move(destroyFuncName), std::move(node),
                                           std::move(destroyMatch), object, std::move(owner), plural};
            }
        }

        // Get all commands. Filter for vulkan commands, not vulkansc
        for(const pugi::xpath_node& command : spec.select_nodes("//commands/command[@api='vulkan' or not(@api)]"))
        {
            std::string createFuncName = command.node().select_node("proto/name/text()").node().value();
            std::smatch createMatch;
            if(std::regex_match(createFuncName, createMatch, createFunc))
            {
                std::string type = command.node().select_node("param[last()]/type/text()").node().value();
                std::string typeBase = std::regex_replace(type, typeBaseStrip, "");
                std::smatch typeSuffixMatch;
                std::string typeSuffix = std::regex_search(type, typeSuffixMatch, suffixPattern) ? typeSuffixMatch[0].str() : "";
                std::string objectName = createMatch[2].str() + createMatch[3].str();

                // Make plural object creation singular
                // TODO: support plural containers?
                pugi::xpath_node countNode = command.node().select_node("param/name[contains(text(),'Count')]");
                bool plural = static_cast<bool>(countNode);
                if(plural)
                    objectName = std::regex_replace(objectName, pluralStrip, "$1");

                auto destroyFunc = destroyCommands.find(type);
                if(destroyFunc == destroyCommands.end())
                {
                    inja::json obj = inja::json::object();
                    obj["name"] = objectName;
                    obj["create"] = createFuncName;
                    obj["failure"] = "No definition destroy function for " + type;
                    handles.push_back(obj);
                    continue;
                }

                pugi::xpath_node createInfoNode = command.node().select_node("param/name[contains(text(),'Info')]");
                if(!createInfoNode)
                {
                    inja::json obj = inja::json::object();
                    obj["name"] = objectName;
                    obj["create"] = createFuncName;
                    obj["failure"] = "Could not find CreateInfo";
                    handles.push_back(obj);
                    continue;
                }

                std::string createInfo = createInfoNode.parent().select_node("type/text()").node().value();
                pugi::xpath_variable_set vars;
                vars.set("createFuncName", createFuncName.c_str());

                inja::json obj = inja::json::object();
                obj["name"] = objectName;
                obj["type"] = type;
                obj["suffix"] = createMatch[3].str();
                obj["owner"] = destroyFunc->second.owner;
                obj["createInfo"] = createInfo;
                obj["create"] = createFuncName;
                obj["createPlural"] = plural;
                obj["destroy"] = destroyFunc->second.name;
                obj["destroyPlural"] = destroyFunc->second.plural;
                obj["failure"] = false;

                pugi::xpath_node extensionCommand = spec.select_node("//extensions/extension/require/command[@name=$createFuncName]", &vars);
                if(extensionCommand)
                    obj["extension"] = extensionCommand.parent().parent().attribute("name").value();
                else
                    obj["extension"] = false;

                handles.push_back(obj);
            }
        }
    }
    data["handles"] = handles;

    // Compute device handles recursively in order to find device functions.
    // This should be flat in the spec :(
    std::unordered_map<std::string_view, std::vector<std::string_view>> handleChildren;
    for(const pugi::xpath_node& handle : spec.select_nodes("//types/type[@category='handle']"))
    {
        std::string_view name = handle.node().child("name").text().get();
        std::string_view parent = handle.node().attribute("parent").value();
        handleChildren[parent].push_back(name);
    }
    const auto handleDFS = [](const auto& addChildren, const auto& handleChildren, auto& handles, std::string_view parent) -> void {
        handles.insert(parent);
        auto children = handleChildren.find(parent);
        if(children != handleChildren.end())
            for(const std::string_view& child : children->second)
                addChildren(addChildren, handleChildren, handles, child);
    };
    std::unordered_set<std::string_view> instanceHandles;
    std::unordered_set<std::string_view> deviceHandles;
    handleDFS(handleDFS, handleChildren, deviceHandles, "VkDevice");
    handleDFS(handleDFS, handleChildren, instanceHandles, "VkInstance");

    // TODO: asymmetry with vkCreateInstance and vkDestroyInstance being in
    // separate tables feels wrong.
    inja::json instanceFunctions = inja::json::array();
    inja::json deviceFunctions = inja::json::array();
    inja::json globalFunctions = inja::json::array();
    for (const pugi::xpath_node& command :
         spec.select_nodes("//commands/command[@api='vulkan' or not(@api)]/proto/..")) {
        std::string_view firstParam =
            command.node().select_node("param[1]/type/text()").node().value();
        std::string_view funcName = command.node().select_node("proto/name/text()").node().value();
        if (instanceHandles.count(firstParam) != 0)
            instanceFunctions.push_back(funcName);
        else if (deviceHandles.count(firstParam) != 0)
            deviceFunctions.push_back(funcName);
        else
            globalFunctions.push_back(funcName);
    }
    data["instance_functions"] = instanceFunctions;
    data["device_functions"] = deviceFunctions;
    data["global_functions"] = globalFunctions;

    try {
        env.render_to(outputFile, templatArghReservedKeyword, data);
    } catch (const inja::RenderError& e) {
        std::cout << "Error rendering template '" << outputFilename << "':\n" << e.what() << "\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
