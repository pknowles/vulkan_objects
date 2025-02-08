// Copyright (c) 2024 Pyarelal Knowles, MIT License

#pragma once

#include <scvk/config.hpp>

namespace scvk {

template <typename T>
concept HasInstance = requires(T t) {
    { t.instance() } -> std::same_as<VkDevice>;
};

template <typename T>
concept HasDevice = requires(T t) {
    { t.device() } -> std::same_as<VkDevice>;
};

template <typename T>
concept HasInstanceTable = requires(T t) {
    t.vki -> std::same_as<InstanceTable>;
};

template <typename T>
concept HasDeviceTable = requires(T t) {
    t.vki -> std::same_as<DeviceTable>;
};

template <typename T>
concept HasAllocationCallbacks = requires(T t) {
    { t.allocationCallbacks() } -> std::same_as<const VkAllocationCallbacks*>;
};

template <typename T>
concept InstanceObjectDestroyer = HasInstance<T> && HasInstanceTable<T> && HasAllocationCallbacks<T>;

template <typename T>
concept DeviceObjectDestroyer = HasInstance<T> && HasDeviceTable<T> && HasAllocationCallbacks<T>;

}
