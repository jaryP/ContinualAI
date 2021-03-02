import os

import setuptools
import codecs


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()


def get_version(rel_path):
    print(rel_path)
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


setuptools.setup(
    name='continual_learning',
    version=get_version("continual_learning/__init__.py"),
    author="Jary Pomponi",
    author_email="jarypomponi@gmail.org",
    description="A base CL framework to speed-up prototyping and testing",
    # long_description=long_description,
    # long_description_content_type="text/markdown",
    url="https://github.com/jaryP/ContinualAI",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 1 - Planning",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    python_requires='>=3.7',
    install_requires=[
        'torch',
        'torchvision',
        'numpy',
        'quadprog',
        'scipy'
    ]
)