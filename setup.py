# This file is part of MetaQuantus.
# MetaQuantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# MetaQuantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with MetaQuantus. If not, see <https://www.gnu.org/licenses/>.

import importlib
from setuptools import setup, find_packages
from sys import version_info
from importlib import util

# Interpret the version of a package depending on if python>=3.8 vs python<3.8:
# Read: https://stackoverflow.com/questions/20180543/how-to-check-version-of-python-modules?rq=1.
if version_info[1] <= 7:
    import pkg_resources

    def version(s: str):
        return pkg_resources.get_distribution(s).version

else:
    from importlib.metadata import version

# Define setup.
setup(
    name="metaquantus",
    version="0.0.1",
    description="MetaQuantus is a XAI performance tool for identifying reliable metrics.",
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    install_requires=[
        "captum>=0.4.1",
        "black==22.10.0",
        "torch>=1.10.1",
        "torchvision>=0.14.1",
        "pandas==1.5.2",
        "quantus>=0.3.2",
    ],
    # extras_require=EXTRAS,
    url="https://github.com/annahedstroem/MetaQuantus",
    author="Anna Hedstrom",
    author_email="hedstroem.anna@gmail.com",
    keywords=[
        "explainable ai",
        "evaluation",
        "xai",
        "machine learning",
        "deep learning",
    ],
    license="GNU LESSER GENERAL PUBLIC LICENSE VERSION 3",
    packages=find_packages(),
    zip_safe=False,
    python_requires=">=3.7",
    include_package_data=True,
)
