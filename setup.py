import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    requirements = fh.readlines()

setuptools.setup(
    name="spnflow",
    version="0.3.1",
    author="loreloc",
    author_email="lorenzoloconte@outlook.it",
    description="Sum-Product Networks and Normalizing Flows for Tractable Density Estimation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/loreloc/spnflow",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=requirements,
)
