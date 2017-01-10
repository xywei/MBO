cmake_minimum_required(VERSION 3.4)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)


# Find packages
FIND_PACKAGE(deal.II 8.3.0 REQUIRED
             HINTS ${deal.II_DIR} ${DEAL_II_DIR} /opt/dealii /usr $ENV{DEAL_II_DIR}
            )
DEAL_II_INITIALIZE_CACHED_VARIABLES()

find_package(Fgt REQUIRED
  HINTS ${FGT_DIR} $ENV{FGT_DIR} ~/apps/fgt 
  )

project(MBOX LANGUAGES CXX)

# Additional include dirs
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/common)

# Make directories for input and output
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/input)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/output)

# Copy input files (mesh etc.) to build directory
file(COPY ${CMAKE_SOURCE_DIR}/input DESTINATION ${CMAKE_BINARY_DIR})

set(MBOX_SOURCE mbox_main.cpp)
add_executable(mbox ${MBOX_SOURCE})
DEAL_II_SETUP_TARGET(mbox)

target_link_libraries(mbox
    PUBLIC
    Fgt::Library-C++
    )

ADD_CUSTOM_TARGET(clear
        COMMAND rm -rf ${CMAKE_BINARY_DIR}/output/*
        COMMAND rm ${CMAKE_BINARY_DIR}/input/*.prm
        COMMENT "Deleted all output stuff and generated parameter files"
        )

ADD_CUSTOM_TARGET(debug
        COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug ${CMAKE_SOURCE_DIR}
        COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
        COMMENT "Switch CMAKE_BUILD_TYPE to Debug"
        )

ADD_CUSTOM_TARGET(release
        COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release ${CMAKE_SOURCE_DIR}
        COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
        COMMENT "Switch CMAKE_BUILD_TYPE to Release"
        )

ADD_CUSTOM_TARGET(run
        COMMAND mbox
        COMMENT "Run solver with ${CMAKE_BUILD_TYPE} configuration"
        )

ADD_CUSTOM_TARGET(param
        COMMAND parameter_gui ${CMAKE_BINARY_DIR}/output/parameters.xml
        )
