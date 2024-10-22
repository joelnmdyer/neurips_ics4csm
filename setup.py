# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
from os.path import abspath, dirname, join

this_dir = abspath(dirname(__file__))
with open(join(this_dir, "LICENSE")) as f:
    license = f.read()

with open(join(this_dir, "README.md"), encoding="utf-8") as file:
    long_description = file.read()

with open(join(this_dir, "requirements.txt")) as f:
    requirements = f.read().split("\n")


setup(
    name="neurips_ics4csm",
    version="1.0.0",
    description="Code for NeurIPS 2024 paper on `Interventionally consistent surrogates for complex simulation models'.",
    url="https://github.com/joelnmdyer/neurips_ics4csm",
    author="Joel Dyer",
    author_email="joel.dyer@cs.ox.ac.uk",
    license="MIT License",
    install_requires=requirements,
    packages=find_packages(exclude=["docs"]),
    include_package_data=True,
)
