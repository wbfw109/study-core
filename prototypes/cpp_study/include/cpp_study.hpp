#pragma once

#include <string>
#include <vector>

#ifdef _WIN32
#define CPP_STUDY_EXPORT __declspec(dllexport)
#else
#define CPP_STUDY_EXPORT
#endif

CPP_STUDY_EXPORT void cpp_study();
CPP_STUDY_EXPORT void cpp_study_print_vector(
    const std::vector<std::string> &strings);
