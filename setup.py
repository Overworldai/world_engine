"""
Build script for world_engine.

On macOS (Apple Silicon), this also compiles the we_kernels Metal/C++ extension
so it ships inside the wheel — no separate `we-kernels` package needed.
On other platforms the extension is skipped and world_engine is pure-Python.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

_EXT_SOURCE_DIR = Path(__file__).parent / "src" / "mlx_metal" / "ext"


# ---------------------------------------------------------------------------
# CMake helpers (derived from mlx.extension, inlined to avoid build-time
# dependency on mlx — MLX_DIR is resolved via `python -m mlx --cmake-dir`).
# ---------------------------------------------------------------------------

class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
            "-DBUILD_SHARED_LIBS=ON",
        ]
        build_args = []

        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        if sys.platform.startswith("darwin"):
            archs = re.findall(r"-arch (\\S+)", os.environ.get("ARCHFLAGS", ""))
            if archs:
                cmake_args += [f"-DCMAKE_OSX_ARCHITECTURES={';'.join(archs)}"]

        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            build_args += [f"-j{os.cpu_count()}"]

        build_temp = Path(self.build_temp) / ext.name
        if not build_temp.exists():
            build_temp.mkdir(parents=True)

        # Point CMake at the installed MLX package so it can find MLXConfig.cmake.
        try:
            import mlx
            os.environ["MLX_DIR"] = str(mlx.__path__[0])
        except ImportError:
            pass  # CMake will fall back to its own search paths

        subprocess.run(
            ["cmake", ext.sourcedir, *cmake_args], cwd=build_temp, check=True
        )
        subprocess.run(
            ["cmake", "--build", ".", *build_args], cwd=build_temp, check=True
        )

    def run(self) -> None:
        super().run()
        if self.inplace:
            for ext in self.extensions:
                if isinstance(ext, CMakeExtension):
                    build_py = self.get_finalized_command("build_py")
                    inplace_file, regular_file = self._get_inplace_equivalent(
                        build_py, ext
                    )
                    inplace_dir = str(Path(inplace_file).parent.resolve())
                    regular_dir = str(Path(regular_file).parent.resolve())
                    self.copy_tree(regular_dir, inplace_dir)


# ---------------------------------------------------------------------------
# Conditional extension: only build we_kernels on macOS arm64.
# ---------------------------------------------------------------------------

ext_modules = []
cmdclass = {}

if sys.platform == "darwin":
    ext_modules = [CMakeExtension("we_kernels._ext", sourcedir=str(_EXT_SOURCE_DIR))]
    cmdclass = {"build_ext": CMakeBuild}

setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
