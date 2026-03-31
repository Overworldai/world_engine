from mlx import extension
from setuptools import setup

setup(
    name="we_kernels",
    version="0.1.0",
    description="World Engine Metal kernels — custom MLX C++ extensions for Apple Silicon",
    ext_modules=[extension.CMakeExtension("we_kernels._ext")],
    cmdclass={"build_ext": extension.CMakeBuild},
    packages=["we_kernels"],
    package_data={"we_kernels": ["*.so", "*.dylib", "*.metallib"]},
)
