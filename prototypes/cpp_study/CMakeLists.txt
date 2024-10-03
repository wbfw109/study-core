cmake_minimum_required(VERSION 3.15)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(cpp_study CXX)

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
endif()

# Decide which package to use.
find_package(OpenCV REQUIRED)
find_package(fmt REQUIRED)
# examples2/examples/libraries/tensorflow-lite/pose-estimation/CMakeLists.txt


include_directories(${PROJECT_SOURCE_DIR}/include)
set(SOURCES
    src/main.cpp
    src/cpp_study.cpp
    src/basic.cpp
    src/brahem.cpp
    src/assignment/Canvas.cpp
    src/assignment/Student.cpp
    src/temp.cpp
)
add_executable(cpp_study ${SOURCES})
target_link_libraries(cpp_study PRIVATE ${CONAN_LIBS} opencv::opencv fmt::fmt)

# install(TARGETS cpp_study DESTINATION "."
#         RUNTIME DESTINATION bin
#         ARCHIVE DESTINATION lib
#         LIBRARY DESTINATION lib
#         )

#[===[
📝 Note that conan 2 autmoatically pass settings from CMakeToolchain 🔗 https://docs.conan.io/2/reference/config_files/settings.html
    : like build_type and compiler.gcc.cppstd corresponding to DCMAKE_BUILD_TYPE and DCMAKE_CXX_STANDARD respectively. + even DCMAKE_CXX_COMPILER
    
    When you run conan install or conan create command, It creates CMakeUserPresets.json.
    And if you follow that "...CMakePresets.json" path, you see written file that indicate CMake configurations, which is automatically created from <conan install ...> command.
    Its infomration is used in command "cmake --preset conan-debug", and "cmake --build --preset conan-debug".
    ➡️ So you only manage confanfile.py and command arguments.
    ⚠️ If you change settings and when use Cmake command revisions are not applied, remove build/ directory. reason may be the Cache
        $shell> rm -rf build/


📝 VSCode's intellisense can be used by export compile_commands.json in CMakeLists.txt and by setting .vscode/settings.json
    🔗 https://stackoverflow.com/a/70616885
]===]