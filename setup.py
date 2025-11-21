import subprocess
from pathlib import Path
from setuptools import setup


def ensure_submodules():
    if Path(".gitmodules").exists():
        try:
            subprocess.run(
                ["git", "submodule", "update", "--init", "--recursive"],
                check=True,
            )
        except Exception as e:
            # Don't hard-fail install if git is missing; just skip
            print(f"Warning: could not update submodules: {e}")


def parse_requirements(path: str):
    path = Path(path)
    requirements = []

    def _parse(file_path: Path):
        for raw in file_path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            # Follow included requirement files
            if line.startswith("-r "):
                included = line.split(maxsplit=1)[1]
                _parse(file_path.parent / included)
                continue

            # Special case: owl-vaes uses this git URL, which actually installs lpips
            if line.startswith("git+https://github.com/shahbuland/PerceptualSimilarity"):
                # Use the published package instead of the bare git URL
                requirements.append("lpips==0.1.4")
                continue

            # For any other VCS lines, you can either ignore or handle later.
            if line.startswith("git+"):
                # For now: skip other git dependencies rather than breaking install_requires
                # (add handling if you need them installed automatically)
                continue

            # Normal requirement line
            requirements.append(line)

    _parse(path)
    return requirements


ensure_submodules()
install_requires = parse_requirements("requirements.txt")

setup(
    name="world_engine",
    version="0.0.1",
    packages=[
        "world_engine",
        "depth_anything_v2",
        "depth_anything_v2.dinov2_layers",
        "depth_anything_v2.util",
        "owl_vaes",
        "owl_wms",
        "owl_wms.models",
        "owl_wms.nn",
    ],
    package_dir={
        "world_engine": "src",

        "depth_anything_v2": "submodules/Depth-Anything-V2/depth_anything_v2",
        "depth_anything_v2.dinov2_layers": "submodules/Depth-Anything-V2/depth_anything_v2/dinov2_layers",

        "owl_vaes": "submodules/owl-wms/owl-vaes/owl_vaes",

        "owl_wms": "submodules/owl-wms/owl_wms",
        "owl_wms.models": "submodules/owl-wms/owl_wms/models",
        "owl_wms.nn": "submodules/owl-wms/owl_wms/nn",
    },
    install_requires=install_requires,
)
