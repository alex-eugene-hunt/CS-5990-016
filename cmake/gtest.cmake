# Download and unpack googletest at configure time
message(STATUS "Getting gtests...")

include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest
    GIT_TAG release-1.12.1
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_GetProperties(googletest)
if (NOT googletest_POPULATED)
    FetchContent_Populate(googletest)
    add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BUILD_DIR})
endif()

# configure_file(cmake/gtest_download.cmake ${PROJECT_SOURCE_DIR}/externals/googletest-download/CMakeLists.txt)
# execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
#   RESULT_VARIABLE result
#   WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/externals/googletest-download
#   OUTPUT_QUIET)
# if(result)
#   message(FATAL_ERROR "CMake step for googletest failed: ${result}")
# endif()
# execute_process(COMMAND ${CMAKE_COMMAND} --build .
#   RESULT_VARIABLE result
#   WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/externals/googletest-download
#   OUTPUT_QUIET)
# if(result)
#   message(FATAL_ERROR "Build step for googletest failed: ${result}")
# endif()

# message(STATUS "gtests downloaded!")

# # Prevent overriding the parent project's compiler/linker
# # settings on Windows
# set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# # Add googletest directly to our build. This defines
# # the gtest and gtest_main targets.
# add_subdirectory(${PROJECT_SOURCE_DIR}/externals/googletest-src
#                  ${PROJECT_SOURCE_DIR}/externals/googletest-build
#                  EXCLUDE_FROM_ALL)

# # The gtest/gtest_main targets carry header search path
# # dependencies automatically when using CMake 2.8.11 or
# # later. Otherwise we have to add them here ourselves.
# if (CMAKE_VERSION VERSION_LESS 2.8.11)
#   include_directories("${gtest_SOURCE_DIR}/include")
# endif()