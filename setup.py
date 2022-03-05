from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="mankey_stats",
    version="0.0.1",
    author="GROUP_A",
    author_email="khalid.nass@student.ie.edu",
    description="A small package for data cleaning library ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dBlueG/mankey_stats",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)