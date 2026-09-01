cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_RUN_MODE "npu" CACHE STRING "Run mode: npu, sim")
set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture")
set(SIMT_CASE "1" CACHE STRING "SIMTVF body: 1=add, 2=long dependency chain")
set(SIMT_THREADS "32" CACHE STRING "SIMTVF threads (M)")
set(DATA_SIZE "256" CACHE STRING "Number of float elements (N)")
set(VF_CALLS "1" CACHE STRING "Consecutive identical SIMTVF calls in the timed region")

if(NOT SIMT_CASE MATCHES "^[12]$")
    message(FATAL_ERROR "SIMT_CASE must be 1 or 2")
endif()
if(SIMT_THREADS LESS 1 OR SIMT_THREADS GREATER 2048)
    message(FATAL_ERROR "SIMT_THREADS must be in [1, 2048]")
endif()
if(DATA_SIZE LESS 1)
    message(FATAL_ERROR "DATA_SIZE must be positive")
endif()
if(NOT VF_CALLS MATCHES "^[12]$")
    message(FATAL_ERROR "VF_CALLS must be 1 or 2")
endif()

find_package(ASC REQUIRED)

project(simtvf_cycle_test LANGUAGES ASC CXX)

add_executable(simtvf_cycle_count
    cycle_count/cycle_count.asc
)

target_compile_definitions(simtvf_cycle_count PRIVATE
    SIMT_CASE=${SIMT_CASE}
    SIMT_THREADS=${SIMT_THREADS}
    DATA_SIZE=${DATA_SIZE}
    VF_CALLS=${VF_CALLS}
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
