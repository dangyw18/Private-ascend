# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------


cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_RUN_MODE "npu" CACHE STRING "Run mode: npu, sim")
set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture: dav-3510")
set(REDUCE_VF_CALLS "0" CACHE STRING
    "Number of consecutive cce3 VF calls: 0 keeps the original cce5 demo; 1 or 2 enables the startup-difference test")

if(NOT REDUCE_VF_CALLS MATCHES "^[012]$")
    message(FATAL_ERROR "REDUCE_VF_CALLS must be 0, 1, or 2")
endif()

find_package(ASC REQUIRED)

project(kernel_samples LANGUAGES ASC CXX)

add_executable(demo
    reduce_demo.asc
)

target_compile_definitions(demo PRIVATE
    REDUCE_VF_CALLS=${REDUCE_VF_CALLS}
)

# CANN 9.1.0-beta.3 的 asc_simd.h 在 #if __NPU_ARCH__ 分支之前就无条件包含
# "instr_impl/npu_arch_2201/utils_impl/utils_impl.h",该文件位于 asc/impl 树下。
# bisheng 对 asc 头文件搜索路径的自动推导依赖 ASCEND_HOME_PATH 的安装布局
# (顶层 asc/ 或 x86_64-linux/asc/),某些布局下不会自动加入 asc/impl 目录,
# 导致编译报 "instr_impl/npu_arch_2201/utils_impl/utils_impl.h file not found"。
# 这里基于 CANN 自带变量显式补全搜索路径,同时覆盖两种布局,不依赖 asc-devkit。
target_include_directories(demo PRIVATE
    ${ASCEND_CANN_PACKAGE_LINUX_PATH}/asc/include
    ${ASCEND_CANN_PACKAGE_LINUX_PATH}/asc/impl/c_api
    ${ASCEND_CANN_PACKAGE_LINUX_PATH}/asc/impl/adv_api
    ${ASCEND_CANN_PACKAGE_LINUX_PATH}/asc/impl/basic_api
    ${ASCEND_CANN_PACKAGE_LINUX_PATH}/asc/impl/simt_api
    ${ASCEND_CANN_PACKAGE_LINUX_PATH}/asc/impl/utils
    $ENV{ASCEND_HOME_PATH}/asc/include
    $ENV{ASCEND_HOME_PATH}/asc/impl/c_api
    $ENV{ASCEND_HOME_PATH}/asc/impl/adv_api
    $ENV{ASCEND_HOME_PATH}/asc/impl/basic_api
    $ENV{ASCEND_HOME_PATH}/asc/impl/simt_api
    $ENV{ASCEND_HOME_PATH}/asc/impl/utils
)

target_link_libraries(demo PRIVATE
    tiling_api
    register
    platform
    m
)

target_compile_options(demo PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
)
