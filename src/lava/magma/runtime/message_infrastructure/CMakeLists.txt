cmake_minimum_required(VERSION 3.5)
project(message_passing)

set (CMAKE_CXX_STANDARD 17)

find_package(pybind11 REQUIRED)

set(ABSTRACT_PORT_SRCS
    "message_infrastructure/csrc/abstract_port_implementation.cc"
)

set(PORT_SRCS
    "message_infrastructure/csrc/ports.cc"
)

set(TRANSFORMER_SRCS
    "message_infrastructure/csrc/transformer.cc"
)

file (GLOB PY_WRAPPER "message_infrastructure/csrc/message_infrastructure_py_wrapper.cc")

set (MULTI_PROC_SRCS
     "message_infrastructure/csrc/multiprocessing.cc"
     "message_infrastructure/csrc/posix_actor.cc"
     "message_infrastructure/csrc/shmem_port.cc")


file (GLOB PY_WRAPPER "message_infrastructure/csrc/message_infrastructure_py_wrapper.cc")

add_library(multi_processing SHARED ${MULTI_PROC_SRCS})
add_library(abstract_port SHARED ${ABSTRACT_PORT_SRCS})
add_library(port SHARED ${PORT_SRCS})
add_library(transformer SHARED ${TRANSFORMER_SRCS})

target_include_directories(multi_processing PUBLIC 
                           ${pybind11_INCLUDE_DIRS}
                           ${PROJECT_SOURCE_DIR}/message_infrastructure/csrc
                           )

pybind11_add_module(MessageInfrastructurePywrapper ${PY_WRAPPER})
target_link_libraries(MessageInfrastructurePywrapper PRIVATE multi_processing)
target_link_libraries(MessageInfrastructurePywrapper PRIVATE abstract_port)
target_link_libraries(MessageInfrastructurePywrapper PRIVATE port)
target_link_libraries(MessageInfrastructurePywrapper PRIVATE transformer)
