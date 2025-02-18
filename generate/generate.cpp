// Copyright (c) 2025 Pyarelal Knowles, MIT License

#include <algorithm>
#include <filesystem>
#include <inja/inja.hpp>
#include <iostream>
#include <pugixml.hpp>
#include <ranges>
#include <regex>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;

struct Command {
    std::string    name;
    pugi::xml_node node;
    std::smatch    match;
    std::string    object;
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

struct SortedStrings {
    std::vector<std::string_view> strings;
    void insert(std::string_view sv) {
        strings.push_back(sv);
        std::ranges::sort(strings);
    }
    bool operator==(const SortedStrings& other) const {
        return std::ranges::equal(strings, other.strings);
    }
};

// Probably terrible, but it just has to work for now.
// Loosely based off https://stackoverflow.com/questions/17016175/c-unordered-map-using-a-custom-class-type-as-the-key
// See also: https://stackoverflow.com/questions/35985960/c-why-is-boosthash-combine-the-best-way-to-combine-hash-values
template <>
struct std::hash<SortedStrings>
{
  std::size_t operator()(const SortedStrings& k) const
  {
      return std::transform_reduce(
          k.strings.begin(), k.strings.end(), size_t{0},
          [](size_t a, size_t b) { return (a ^ (b << 1)) >> 1; }, std::hash<std::string_view>{});
  }
};

// *sigh*. almost
auto split(std::string_view str, char delim)
{
    // &*rng.begin() is UB if the range is empty, so we need
    // to check before using it
    return str | std::views::split(delim) | std::views::transform([](auto&& rng) {
               const auto size = std::ranges::distance(rng);
               return size ? std::string_view(&*rng.begin(), size) : std::string_view();
           });
}

std::unordered_map<std::string_view, std::string_view> makeCommandRootParents(const pugi::xml_document& spec)
{
    // Compute device handles recursively in order to find device functions.
    // This should be flat in the spec :(
    // NOTE: in this code, instance handles are all device handles since the
    // "parent" of a device is an instance
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

    std::unordered_map<std::string_view, std::string_view> commandRootParents;
    for (const pugi::xpath_node& command :
         spec.select_nodes("//commands/command/proto/..")) {
        std::string_view firstParam =
            command.node().select_node("param[1]/type/text()").node().value();
        auto commandName = command.node().select_node("proto/name/text()").node().value();
        if (deviceHandles.count(firstParam) != 0)
            commandRootParents[commandName] = "VkDevice";
        else if (instanceHandles.count(firstParam) != 0)
            commandRootParents[commandName] = "VkInstance";
        else
            commandRootParents[commandName] = "";
    }
    for (const pugi::xpath_node& command : spec.select_nodes("//commands/command[@alias]")) {
        std::string_view name = command.node().attribute("name").value();
        std::string_view alias = command.node().attribute("alias").value();
        commandRootParents[name] = commandRootParents[alias];
    }

    // Override the loaders since they are used to populate each table and must
    // be loaded in the parent table.
    commandRootParents["vkGetDeviceProcAddr"] = "VkInstance";
    commandRootParents["vkGetInstanceProcAddr"] = "";

    return commandRootParents;
}

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
        std::cout << "Failed to open output file " << outputFilename << "\n";
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

    env.add_callback("substr", [](inja::Arguments& args) {
        auto s = args.at(0)->get<std::string>();
        auto a = args.at(1)->get<ptrdiff_t>();
        auto b = args.size() > 2 ? args.at(2)->get<ptrdiff_t>() : ptrdiff_t(s.size());
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
    } catch ( const inja::ParserError& e ){
        std::cout << templateFilename.string() << ":" << e.location.line << ":" << e.location.column << ": error: " << e.message << "\n";
        return EXIT_FAILURE;
    }

    // Only generate for vulkan, not vulkansc
    auto apiAllowed = [](std::string_view apis) {
        if (apis.empty())
            return true;
        for (auto api : split(apis, ','))
            if (api == "vulkan")
                return true;
        return false;
    };

    std::unordered_map<std::string_view, std::string_view> commandRootParents = makeCommandRootParents(spec);

    inja::json data;

    inja::json handles = inja::json::array();
    {
        std::regex createFunc("vk(Create|Allocate)(.*)([A-Z]{2,})?");
        std::regex destroyFunc("vk(Destroy|Free)(.*)([A-Z]{2,})?");
        std::regex typeBaseStrip("^Vk|[A-Z]{2,}$");
        std::regex suffixPattern("[A-Z]{2,}$");
        std::regex pluralStrip("(.*)(ies|es|s)$");

        std::unordered_map<std::string, Command> destroyCommands;
        for(const pugi::xpath_node& command : spec.select_nodes("//commands/command"))
        {
            pugi::xml_node node = command.node();
            if(!apiAllowed(node.attribute("api").value()))
                continue;
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
                destroyCommands[object] = {std::move(destroyFuncName), std::move(node),
                                           std::move(destroyMatch), object, plural};
            }
        }

        // Get all commands. Filter for vulkan commands, not vulkansc
        for(const pugi::xpath_node& command : spec.select_nodes("//commands/command"))
        {
            if(!apiAllowed(command.node().attribute("api").value()))
                continue;
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
                obj["owner"] = commandRootParents[destroyFunc->second.name];
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

    std::set<std::string_view> types; // required types only, filtered by apiAllowed()
    std::set<std::string_view> commands; // required commands only, filtered by apiAllowed()
    std::unordered_map<std::string_view, SortedStrings> typesRequiredBy;
    std::unordered_map<std::string_view, SortedStrings> commandsRequiredBy;
    for (const pugi::xpath_node& featureNode :
         spec.select_nodes("//feature/require/..")) {
        std::string_view feature = featureNode.node().attribute("name").value();
        if(!apiAllowed(featureNode.node().attribute("api").value()))
            continue;
        for (const pugi::xpath_node& typeNode :
             spec.select_nodes((std::string("//feature[@name='") + std::string(feature) + "']/require/type/@name")
                 .c_str())) {
            std::string_view type = typeNode.attribute().value();
            types.insert(type);
            typesRequiredBy[type].insert(feature);
        }
        for (const pugi::xpath_node& commandNode :
             spec.select_nodes((std::string("//feature[@name='") + std::string(feature) + "']/require/command/@name")
                 .c_str())) {
            std::string_view command = commandNode.attribute().value();
            commands.insert(command);
            commandsRequiredBy[command].insert(feature);
        }
    }

    inja::json platformTypes = inja::json::array();
    inja::json platformCommands = inja::json::array();
    std::unordered_set<std::string_view> platformExtensions;
    for (const pugi::xpath_node& extensionNode :
         spec.select_nodes("//extensions/extension/require/..")) {
        std::string_view extension = extensionNode.node().attribute("name").value();
        std::string_view promotedto = extensionNode.node().attribute("promotedto").value();
        std::string_view platform = extensionNode.node().attribute("platform").value();
        std::string_view requiredby = promotedto.empty() ? extension : promotedto;
        if(!platform.empty())
            platformExtensions.insert(extension);

        // The extension may be limited to specific APIs
        if(!apiAllowed(extensionNode.node().attribute("supported").value()))
            continue;

        for (const pugi::xpath_node& typeNode :
             spec.select_nodes((std::string("//extensions/extension[@name='") +
                                std::string(extension) + "']/require/type/@name")
                                   .c_str())) {

            // The <require> tag may be limited to specific APIs
            if(!apiAllowed(typeNode.parent().parent().attribute("api").value()))
                continue;

            std::string_view type = typeNode.attribute().value();
            types.insert(type);
            typesRequiredBy[type].insert(requiredby);
            if(!platform.empty())
                platformTypes.push_back(type);
        }
        for (const pugi::xpath_node& commandNode :
             spec.select_nodes((std::string("//extensions/extension[@name='") +
                                std::string(extension) + "']/require/command/@name")
                                   .c_str())) {

            // The <require> tag may be limited to specific APIs
            if(!apiAllowed(commandNode.parent().parent().attribute("api").value()))
                continue;

            std::string_view command = commandNode.attribute().value();
            commands.insert(command);
            commandsRequiredBy[command].insert(requiredby);
            if(!platform.empty())
                platformCommands.push_back(command);
        }
    }

    std::unordered_map<SortedStrings, std::vector<std::string_view>> extensionGroupTypesMap;
    for(auto& [command, requiredBy] : typesRequiredBy)
        extensionGroupTypesMap[requiredBy].push_back(command);
    assert(extensionGroupTypesMap.count(SortedStrings()) == 0); // everything should have a requirement

    std::unordered_map<SortedStrings, std::vector<std::string_view>> extensionGroupCommandsMap;
    for(auto& [command, requiredBy] : commandsRequiredBy)
        extensionGroupCommandsMap[requiredBy].push_back(command);
    assert(extensionGroupCommandsMap.count(SortedStrings()) == 0); // everything should have a requirement

    inja::json deviceCommands = inja::json::array();
    inja::json instanceCommands = inja::json::array();
    inja::json globalCommands = inja::json::array();
    for (auto& command : commands) {
        // It's worse than I thought. You can't load the destroy functions
        // without an instance! *double-picard-facepalm*
        #if 0
        // Hack for vkCreateDevice symmetry; weird to load destroy call after construction
        if(command == "vkDestroyDevice")
        {
            instanceCommands.push_back(command);
            continue;
        }

        // Hack for vkCreateInstance symmetry; weird to load destroy call after construction
        if(command == "vkDestroyInstance")
        {
            globalCommands.push_back(command);
            continue;
        }
        #endif

        if (commandRootParents[command] == "VkDevice")
            deviceCommands.push_back(command);
        else if (commandRootParents[command] == "VkInstance")
            instanceCommands.push_back(command);
        else
            globalCommands.push_back(command);
    }

    inja::json extensionGroupTypes = inja::json::array();
    for(auto& [extensions, types] : extensionGroupTypesMap)
    {
        inja::json extensionGroup;
        extensionGroup["extensions"] = extensions.strings;
        extensionGroup["types"] = types;
        extensionGroupTypes.push_back(extensionGroup);
    }

    inja::json extensionGroupCommands = inja::json::array();
    for(auto& [extensions, commands] : extensionGroupCommandsMap)
    {
        inja::json extensionGroup;
        extensionGroup["extensions"] = extensions.strings;
        extensionGroup["commands"] = commands;

        // Compute hasPlatform
        // TODO: make faster?
        extensionGroup["hasPlatform"] = false;
        for (auto& ext : extensions.strings)
        {
            if (platformExtensions.count(ext))
            {
                extensionGroup["hasPlatform"] = true;
                break;
            }
        }

        // Compute parent objects
        std::unordered_set<std::string_view> parentObjects;
        for(auto& command : commands)
            parentObjects.insert(commandRootParents[command]);
        extensionGroup["parents"] = parentObjects;

        extensionGroupCommands.push_back(extensionGroup);
    }

    data["types"] = types;
    data["instance_commands"] = instanceCommands;
    data["device_commands"] = deviceCommands;
    data["global_commands"] = globalCommands;
    data["platform_types"] = platformTypes;
    data["platform_commands"] = platformCommands;
    data["command_parents"] = commandRootParents;
    data["extension_group_types"] = extensionGroupTypes;
    data["extension_group_commands"] = extensionGroupCommands;
    data["template_filename"] = templateFilename.filename().generic_string();

    try {
        env.render_to(outputFile, templatArghReservedKeyword, data);
    } catch (const inja::RenderError& e) {
        std::cout << templateFilename.string() << ":" << e.location.line << ":" << e.location.column << ": error: " << e.message << "\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
