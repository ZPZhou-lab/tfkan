import setuptools

# Load the long_description from README.md
with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

# load the requirements from requirements.txt
with open("requirements.txt", "r", encoding="utf8") as fh:
    requirements = fh.read().splitlines()

setuptools.setup(
    name="tfkan",
    version="0.1.0",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    author="Xavier Camel",
    description="A Tensorflow implementation of KANs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent'
    ],
    python_requires='>=3.7',
)