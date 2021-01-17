"""Our setup script."""

import glob
from setuptools import setup
import versioneer

setup(
    name="torchdms",
    description="🔥 Tools for analyzing deep mutational scanning data using PyTorch 🔥",
    url="http://github.com/matsengrp/torchdms",
    author="Matsen group",
    author_email="ematsen@gmail.com",
    license="MIT",
    packages=["torchdms"],
    package_data={"torchdms": ["data/*"]},
    scripts=glob.glob("torchdms/scripts/*.sh"),
    entry_points={"console_scripts": ["tdms=torchdms.cli:cli"]},
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    install_requires=[
        "click",
        "click-config-file",
        "dms_variants",
        "matplotlib",
        "plotnine",
        "pytest",
        "scipy",
        "sphinx",
        "sphinx-autodoc-typehints",
        "sphinx-click",
        "sphinx-rtd-theme",
        "statsmodels==0.11.0",
        "torch==1.4.0",
        "versioneer",
    ],
)
