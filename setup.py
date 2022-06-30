# This file is part of CNQR which is released under the Apache License 2.0.
# See file LICENSE in the root directory or https://opensource.org/licenses/Apache-2.0 for full license details.

"""Setup script for CNQR."""

import os
import setuptools
from setuptools import find_packages
from setuptools import setup


folder = os.path.dirname(__file__)
version_path = os.path.join(folder, "cnqr", "version.py")

__version__ = None
with open(version_path) as f:
  exec(f.read(), globals())

req_path = os.path.join(folder, "requirements.txt")
install_requires = []
if os.path.exists(req_path):
  with open(req_path) as fp:
    install_requires = [line.strip() for line in fp]

readme_path = os.path.join(folder, "README.md")
readme_contents = ""
if os.path.exists(readme_path):
  with open(readme_path) as fp:
    readme_contents = fp.read().strip()


setuptools.setup(
    name="cnqr",
    version=__version__,
    author=", ".join(["Louis BETHUNE", "Yannick PRUDENT"]),
    author_email=", ".join(
        [
            "louis.bethune@univ-toulouse.fr",
            "yannick.prudent@irt-saintexupery.com",
        ]
    ),
    description="Jax implementation for Certified Neural Quantile Regression",
    long_description=readme_contents,
    long_description_content_type="text/markdown",
    url="https://github.com/Algue-Rythme/CertifiedQuantileRegression",
    license="Apache V2.0",
    packages=find_packages(),
    package_data={},
    install_requires=install_requires,
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
