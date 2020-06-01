import glob
from setuptools import setup

setup(
    name="torchdms",
    version="0.0.0",
    description="🔥 Tools for analyzing deep mutational scanning data using PyTorch 🔥",
    url="http://github.com/matsengrp/torchdms",
    author="Matsen group",
    author_email="ematsen@gmail.com",
    license="MIT",
    packages=["torchdms"],
    package_data={"torchdms": ["data/*"]},
    scripts=glob.glob("torchdms/scripts/*.sh"),
    entry_points={"console_scripts": ["tdms=torchdms.cli:cli"]},
)
