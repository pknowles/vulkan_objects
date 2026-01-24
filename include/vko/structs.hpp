// Copyright (c) 2025 Pyarelal Knowles, MIT License
#pragma once

#include <utility>
#include <vko/gen_structures.hpp>

namespace vko {

template <class StructType, class... Args>
StructType make(Args&&... args) {
    return StructType{struct_traits<StructType>::sType, std::forward<Args>(args)...};
}

} // namespace vko
