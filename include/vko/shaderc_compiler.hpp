// Copyright (c) 2025 Pyarelal Knowles, MIT License
#pragma once

/*
 * Minimal utilities for shaderc compilation with exception-based error handling.
 *
 * Example usage:
 *
 *   ::shaderc::Compiler compiler;
 *   ::shaderc::CompileOptions options;
 *
 *   // Configure target environment and SPIR-V version
 *   options.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_3);
 *   options.SetTargetSpirv(shaderc_spirv_version_1_6);
 *
 *   // Enable debug info (optional)
 *   #ifndef NDEBUG
 *   options.SetGenerateDebugInfo();
 *   #endif
 *
 *   // Set up include directories
 *   options.SetIncluder(std::make_unique<vko::shaderc::FileIncluder>(
 *       std::vector{std::filesystem::path("shaders"), std::filesystem::path("include")}));
 *
 *   // Compile shader
 *   auto result = compiler.CompileGlslToSpv(source, shaderc_glsl_vertex_shader,
 *                                            "shader.vert", options);
 *
 *   // Print errors/warnings if compilation failed (optional, for debugging)
 *   vko::shaderc::printErrors(result);
 *
 *   // Check for errors (throws on failure)
 *   vko::shaderc::check(result);
 *
 *   // Use result (e.g., create VkShaderModule)
 *   std::span<const uint32_t> spirv{result.begin(), result.end()};
 */

#include <filesystem>
#include <fstream>
#include <print>
#include <shaderc/shaderc.hpp>
#include <span>
#include <string>
#include <vector>
#include <vko/exceptions.hpp>

namespace vko {
namespace shaderc {

inline const char* to_string(shaderc_compilation_status status) {
    switch (status) {
    case shaderc_compilation_status_success:
        return "success";
    case shaderc_compilation_status_invalid_stage:
        return "invalid stage";
    case shaderc_compilation_status_compilation_error:
        return "compilation error";
    case shaderc_compilation_status_internal_error:
        return "internal error";
    case shaderc_compilation_status_null_result_object:
        return "null result object";
    case shaderc_compilation_status_invalid_assembly:
        return "invalid assembly";
    case shaderc_compilation_status_validation_error:
        return "validation error";
    case shaderc_compilation_status_transformation_error:
        return "transformation error";
    case shaderc_compilation_status_configuration_error:
        return "configuration error";
    default:
        return "unknown error";
    }
}

template <typename T>
bool succeededOrPrintErrors(const std::filesystem::path&           path,
                            const ::shaderc::CompilationResult<T>& result, FILE* stream = stderr) {
    if (result.GetCompilationStatus() == ::shaderc_compilation_status_success)
        return true;
    std::println(stream, "Shaderc compilation failed: %s: %s\n%s\n", path.string().c_str(),
                 to_string(result.GetCompilationStatus()), result.GetErrorMessage().c_str());
    return false;
}

template <typename T>
inline void check(::shaderc::CompilationResult<T>& result) {
    auto status = result.GetCompilationStatus();
    if (status != ::shaderc_compilation_status_success) {
        throw vko::Exception("Shaderc compilation failed: " + std::string(to_string(status)));
    }
}

// GLSL to SPIR-V binary. Not sure who would use anything else.
class SpirvBinary {
public:
    // Convenience constructor that prints errors and throws on failure
    SpirvBinary(::shaderc::Compiler& compiler, std::string_view sourceText,
                shaderc_shader_kind shader_kind, std::string inputFileName,
                std::string entryPointName, const ::shaderc::CompileOptions& options)
        : m_result(compiler.CompileGlslToSpv(sourceText.data(), sourceText.size(), shader_kind,
                                             inputFileName.c_str(), entryPointName.c_str(),
                                             options)) {
        if (!succeededOrPrintErrors(inputFileName, m_result))
            check(m_result);
    }

    // Optional failure function to do something with the error messages before throwing
    template <class FailFn>
    SpirvBinary(::shaderc::Compiler& compiler, std::string_view sourceText,
                shaderc_shader_kind shader_kind, std::string inputFileName,
                std::string entryPointName, const ::shaderc::CompileOptions& options, FailFn failFn)
        : m_result(compiler.CompileGlslToSpv(sourceText.data(), sourceText.size(), shader_kind,
                                             inputFileName.c_str(), entryPointName.c_str(),
                                             options)) {
        if (m_result.GetCompilationStatus() != ::shaderc_compilation_status_success)
            failFn(m_result);
        check(m_result);
    }
    std::span<const uint32_t> span() const { return {m_result.cbegin(), m_result.cend()}; }
    operator std::span<const uint32_t>() const { return span(); }

private:
    ::shaderc::SpvCompilationResult m_result;
};

// Simple file-based includer that searches a list of include directories
class FileIncluder : public ::shaderc::CompileOptions::IncluderInterface {
public:
    FileIncluder(std::span<const std::filesystem::path> includeDirs) {
        m_includeDirs.reserve(includeDirs.size());
        for (const auto& dir : includeDirs) {
            try {
                if (std::filesystem::exists(dir)) {
                    m_includeDirs.push_back(std::filesystem::canonical(dir));
                } else {
                    m_includeDirs.push_back(std::filesystem::absolute(dir));
                }
            } catch (...) {
                m_includeDirs.push_back(std::filesystem::absolute(dir));
            }
        }
    }

    shaderc_include_result* GetInclude(const char* requested_source, shaderc_include_type type,
                                       const char* requesting_source,
                                       size_t /* include_depth */) override {
        std::filesystem::path requested(requested_source);
        std::filesystem::path requesting(requesting_source ? requesting_source : "");

        // Try relative to requesting source first
        if (!requesting.empty() && type == shaderc_include_type_relative) {
            auto candidate = requesting.parent_path() / requested;
            if (std::filesystem::exists(candidate)) {
                return makeResult(candidate);
            }
        }

        // Search include directories
        for (const auto& dir : m_includeDirs) {
            auto candidate = dir / requested;
            if (std::filesystem::exists(candidate)) {
                return makeResult(candidate);
            }
        }

        // Failed to find
        auto* data          = new IncludeData;
        data->content       = "Could not find include file: " + std::string(requested_source);
        auto* result        = new shaderc_include_result;
        result->source_name = nullptr;
        result->source_name_length = 0;
        result->content            = data->content.c_str();
        result->content_length     = data->content.size();
        result->user_data          = data;
        return result;
    }

    void ReleaseInclude(shaderc_include_result* data) override {
        if (data && data->user_data) {
            delete static_cast<IncludeData*>(data->user_data);
        }
        delete data;
    }

private:
    struct IncludeData {
        std::string sourceName;
        std::string content;
    };

    shaderc_include_result* makeResult(const std::filesystem::path& path) {
        std::ifstream file(path);
        if (!file.is_open()) {
            auto* data                 = new IncludeData;
            data->content              = "Failed to open file: " + path.string();
            auto* result               = new shaderc_include_result;
            result->source_name        = nullptr;
            result->source_name_length = 0;
            result->content            = data->content.c_str();
            result->content_length     = data->content.size();
            result->user_data          = data;
            return result;
        }

        std::string content((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());
        std::string sourceName;
        try {
            sourceName = std::filesystem::canonical(path).string();
        } catch (...) {
            sourceName = std::filesystem::absolute(path).string();
        }

        auto* data                 = new IncludeData{std::move(sourceName), std::move(content)};
        auto* result               = new shaderc_include_result;
        result->source_name        = data->sourceName.c_str();
        result->source_name_length = data->sourceName.size();
        result->content            = data->content.c_str();
        result->content_length     = data->content.size();
        result->user_data          = data;
        return result;
    }

    std::vector<std::filesystem::path> m_includeDirs;
};

} // namespace shaderc
} // namespace vko