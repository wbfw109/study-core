import os

from conan import ConanFile
from conan.tools.build import check_max_cppstd, check_min_cppstd
from conan.tools.cmake import CMake, CMakeDeps, CMakeToolchain, cmake_layout
from conan.tools.files import copy


class cpp_studyRecipe(ConanFile):
    name = "cpp_study"
    version = "0.1"
    package_type = "application"

    # Optional metadata
    license = "Apache License 2.0"
    author = "wbfw109v2@gmail.ocm"
    url = "https://github.com/abcde111112/intel-edge-academy-6"
    description = "Cpp study"
    topics = ("cpp", "opencv", "vision")

    # Binary configuration
    settings = "os", "compiler", "build_type", "arch"

    options = {"shared": [True, False], "fPIC": [True, False]}

    default_options = {
        "shared": False,
        # "fPIC": True}
        "fPIC": False,
    }

    # Sources are located in the same place as this recipe, copy them to the recipe
    exports_sources = "CMakeLists.txt", "src/*"

    def config_options(self):
        if self.settings.os == "Windows":
            del self.options.fPIC

    def configure(self):
        # ⚓📍 settings.yml ; https://docs.conan.io/2/reference/config_files/settings.html
        # ⚓💡 profiles ; https://docs.conan.io/2/reference/config_files/profiles.html

        if self.options.shared:
            # If os=Windows, fPIC will have been removed in config_options()
            # use rm_safe to avoid double delete errors
            self.options.rm_safe("fPIC")

    def layout(self):
        cmake_layout(self)

    def requirements(self):
        self.requires("opencv/[^4.10.0]")
        self.requires("fmt/[^11.0.2]")
        # self.requires("pulseaudio/[^14.0]")

        self.test_requires("gtest/1.15.0")

        # self.requires("libmp3lame/[^3.100]")

    def build_requirements(self):
        self.tool_requires("cmake/[^3.30.1]")
        self.tool_requires("ninja/[^1.12.1]")

    def validate(self):
        check_min_cppstd(self, "23")
        check_max_cppstd(self, "23")

    # https://stackoverflow.com/questions/71548337/how-to-choose-ninja-as-cmake-generator-with-conan
    def generate(self):
        deps = CMakeDeps(self)
        deps.generate()
        tc = CMakeToolchain(self)
        tc.cache_variables["CMAKE_C_COMPILER_LAUNCHER"] = "ccache"
        tc.cache_variables["CMAKE_CXX_COMPILER_LAUNCHER"] = "ccache"
        tc.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        cmake = CMake(self)
        cmake.install()
        copy(
            self,
            "LICENSE",
            src=self.source_folder,
            dst=os.path.join(self.package_folder, "licenses"),
        )


# https://gitlab.kitware.com/cmake/cmake/-/issues/20705
# https://www.google.com/search?q=CMAKE_CXX_FLAGS+C%2B%2B+%EC%84%A4%EC%A0%95+x86_64-w64-mingw3&sca_esv=1939e291f41ef3b2&sca_upv=1&ei=ALLNZvjRPKmg0-kPtr7X2Qg&ved=0ahUKEwj40f7KgpWIAxUp0DQHHTbfNYsQ4dUDCA8&uact=5&oq=CMAKE_CXX_FLAGS+C%2B%2B+%EC%84%A4%EC%A0%95+x86_64-w64-mingw3&gs_lp=Egxnd3Mtd2l6LXNlcnAiLENNQUtFX0NYWF9GTEFHUyBDKysg7ISk7KCVIHg4Nl82NC13NjQtbWluZ3czMggQABiABBiiBDIIEAAYgAQYogQyCBAAGIAEGKIEMggQABiABBiiBDIIEAAYgAQYogRIsixQzw1Y0CpwAngAkAEAmAGzAaAB1QaqAQMwLja4AQPIAQD4AQH4AQKYAgSgAvYEwgIFECEYoAGYAwCIBgGSBwMwLjSgB7EQ&sclient=gws-wiz-serp
# https://stackoverflow.com/questions/58900537/building-a-simple-c-project-on-windows-using-cmake-and-clang
