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
    python_requires=">=3.8,<3.10",
    install_requires=[
        "click",
        "click-config-file",
        "binarymap",
        "matplotlib==3.3.1",
        "pandas==1.4.2",
        "plotnine==0.8.0",
        "scipy",
        "torch",
    ],
)
