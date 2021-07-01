import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    requirements = fh.readlines()

setuptools.setup(
    name="deeprob-kit",
    version="0.6.1",
    author="loreloc",
    author_email="lorenzoloconte@outlook.it",
    description="Python library for Deep Probabilistic Modeling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/loreloc/deeprob-kit",
    packages=setuptools.find_packages(exclude=['test', 'experiments']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=requirements,
)
