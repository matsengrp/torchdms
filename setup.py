from setuptools import setup

setup(
    name="torchdms",
    version="0.0.0",
    description="Tools for analyzing DMS data using PyTorch",
    url="http://github.com/matsengrp/torchdms",
    author="Matsen group",
    author_email="ematsen@gmail.com",
    license="MIT",
    packages=["torchdms"],
    zip_safe=False,
    entry_points={"console_scripts": ["tdms=torchdms.cli:cli"]},
)
