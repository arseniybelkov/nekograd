import runpy
from pathlib import Path

from setuptools import find_packages, setup


with open("requirements.txt", encoding="utf-8") as file:
    requirements = file.read().splitlines()

with open("README.md", encoding="utf-8") as file:
    long_description = file.read()

root = Path(__file__).resolve().parent
folder = root / "nekograd"
version = runpy.run_path(folder / "__version__.py")["__version__"]

setup(
    name="nekograd",
    version=version,
    license="MIT",
    url="https://github.com/arseniybelkov/nekograd",
    install_requires=requirements,
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.7",
)
