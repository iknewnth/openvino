# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

file (GLOB SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file (GLOB HDR ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

find_package(HddlUnite QUIET)
if(HddlUnite_FOUND)
    # test hddl2_params.hpp
    if(EXISTS ${KMBPLUGIN_HOME}/include/hddl2/hddl2_params.hpp)
        set(HDDL2_PARAMS_DIR ${KMBPLUGIN_HOME}/include/)
        add_definitions(-DUSE_REMOTE_MEM)
        set(HDDL2_DEP "HddlUnite::HddlUnite")
    else()
        message(WARNING "hddl2_params.hpp could not find. Preallocating memory in remote device feature is disabled.")
        remove_definitions(-DUSE_REMOTE_MEM)
    endif()
endif()

ie_add_sample(NAME benchmark_app
              SOURCES ${SRC}
              HEADERS ${HDR}
              DEPENDENCIES format_reader ie_samples_utils ${HDDL2_DEP}
              OPENCV_DEPENDENCIES imgcodecs
              INCLUDE_DIRECTORIES ${HDDL2_PARAMS_DIR})

