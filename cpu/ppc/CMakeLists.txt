include_directories("${PROJECT_SOURCE_DIR}"
                    "${PROJECT_SOURCE_DIR}/thirdparty/loguru/")

# DPPC_JIT_ARCH_DIR is picked in the top level CMakeLists, before this file is
# added, so this target and the bench1 one agree on which emitter builds
file(GLOB SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
                  "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
                  "${CMAKE_CURRENT_SOURCE_DIR}/jit/*.cpp"
                  "${CMAKE_CURRENT_SOURCE_DIR}/jit/*.h"
                  "${CMAKE_CURRENT_SOURCE_DIR}/jit/threaded/*.cpp"
    )

# jit/host holds the only code that asks the operating system for anything,
# so it builds under the same condition as the emitter it serves
if (DPPC_JIT_ARCH_DIR)
    file(GLOB JIT_ARCH_SOURCES
         "${CMAKE_CURRENT_SOURCE_DIR}/jit/host/*.cpp"
         "${CMAKE_CURRENT_SOURCE_DIR}/jit/host/*.h"
         "${CMAKE_CURRENT_SOURCE_DIR}/jit/${DPPC_JIT_ARCH_DIR}/*.cpp"
         "${CMAKE_CURRENT_SOURCE_DIR}/jit/${DPPC_JIT_ARCH_DIR}/*.h")
    list(APPEND SOURCES ${JIT_ARCH_SOURCES})
endif()

add_library(cpu_ppc OBJECT ${SOURCES})
