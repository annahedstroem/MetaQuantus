# This file is part of MetaQuantus.
# MetaQuantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# MetaQuantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with MetaQuantus. If not, see <https://www.gnu.org/licenses/>.

from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

# Define setup.
setup(
    name="metaquantus",
    version="0.0.5",
    description="MetaQuantus is a XAI performance tool for identifying reliable metrics.",
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    install_requires=required,
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
    python_requires=">=3.8",
    include_package_data=True,
)
