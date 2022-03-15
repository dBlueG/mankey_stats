from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()


INSTALL_REQUIRES = [
      'numpy',
      'pandas',
      'scikit-learn',
      'scipy',
      'plotly',
      'matplotlib'
]

test_requirements = ['pytest','Sphinx','rinohtype']
  
setup(
    name="mankey_stats",
    version="0.0.7",
    author="IE GROUP_A",
    author_email="khalid.nass@student.ie.edu",
    description="Clean and transform data for ML binary classification with ease",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dBlueG/mankey_stats",
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    tests_require = test_requirements, 
    keywords=['python', 'pandas', 'classification'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education", 
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ],
    python_requires = ">=3.6"
)