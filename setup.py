#!/usr/bin/env python

import codecs
import os

from setuptools import find_packages
from skbuild import setup

HERE = os.path.dirname(os.path.realpath(__file__))


def read(*parts):
    with codecs.open(os.path.join(HERE, *parts), "rb", "utf-8") as f:
        return f.read()


setup(
    name="jax-finufft",
    author="Dan Foreman-Mackey",
    author_email="foreman.mackey@gmail.com",
    url="https://github.com/dfm/jax-finufft",
    license="MIT",
    description="Unofficial JAX bindings for finufft",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    packages=find_packages("src"),
    package_dir={"": "src"},
    include_package_data=True,
    python_requires=">=3.6",
    install_requires=["jax", "jaxlib"],
    extras_require={"test": ["pytest"]},
    cmake_install_dir="src/jax_finufft",
)
