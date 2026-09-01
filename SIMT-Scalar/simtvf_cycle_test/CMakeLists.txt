cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_RUN_MODE "npu" CACHE STRING "Run mode: npu, sim")
set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture")
set(SIMT_THREADS "32" CACHE STRING "Threads launched by the SIMTVF probe")

find_package(ASC REQUIRED)

project(simtvf_cycle_test LANGUAGES ASC CXX)

add_executable(simtvf_cycle_count
    cycle_count/cycle_count.asc
)

target_compile_definitions(simtvf_cycle_count PRIVATE
    SIMT_THREADS=${SIMT_THREADS}
)

target_include_directories(simtvf_cycle_count PRIVATE
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

target_link_libraries(simtvf_cycle_count PRIVATE
    tiling_api
    register
    platform
    m
)

target_compile_options(simtvf_cycle_count PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
)
