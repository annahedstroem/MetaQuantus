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

# Define extras.
EXTRAS = {}
EXTRAS["torch"] = (
    ["torch==1.10.1", "torchvision==0.11.2"]
    if not (util.find_spec("torch") and version("torch") >= "1.2")
    else []
)
EXTRAS["tensorflow"] = (
    ["tensorflow==2.6.2"]
    if not (util.find_spec("tensorflow") and version("tensorflow") >= "2.0")
    else []
)
EXTRAS["captum"] = (
    (EXTRAS["torch"] + ["captum==0.4.1"]) if not util.find_spec("captum") else []
)
EXTRAS["tf-explain"] = (
    (EXTRAS["tensorflow"] + ["tf-explain==0.3.1"])
    if not util.find_spec("tf-explain")
    else []
)
EXTRAS["zennit"] = (
    (EXTRAS["torch"] + ["zennit==0.4.5"]) if not util.find_spec("zennit") else []
)

# Define setup.
setup(
    name="MetaQuantus",
    version="0.0.1",
    description="MetaMetaQuantus is a toolkit to estimate performance of evaluator of explanation quality.",
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    install_requires=[
        "MetaQuantus"
    ],
    extras_require=EXTRAS,
    url="http://github.com/understandable-machine-intelligence-lab/MetaQuantus",
    author="Anna Hedstrom",
    author_email="hedstroem.anna@gmail.com",
    keywords=["explainable ai", "evaluation", "xai", "machine learning", "deep learning"],
    license="GNU LESSER GENERAL PUBLIC LICENSE VERSION 3",
    packages=find_packages(),
    zip_safe=False,
    python_requires=">=3.7",
    include_package_data=True,
)
