"""NFFLR: Neural Force Field Learning Library

https://jarvis.nist.gov.
"""

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nfflr",
    version="0.1.0",
    author="Brian DeCost, Kamal Choudhary",
    author_email="brian.decost@nist.gov",
    description="neural force field learning toolkit",
    install_requires=[
        "numpy>=1.19.5",
        "scipy>=1.6.1",
        "jarvis-tools>=2021.07.19",
        "torch",
        "dgl>=0.6.0",
        "scikit-learn>=0.22.2",
        "matplotlib>=3.4.1",
        "tqdm>=4.60.0",
        "pandas>=1.2.3",
        "pytorch-ignite",
        "pydantic>=1.8.1",
        "flake8>=3.9.1",
        "pycodestyle>=2.7.0",
        "pydocstyle>=6.0.0",
        "pyparsing>=2.2.1,<3",
        "ase",
    ],
    package_data={},
    scripts=[],
    entry_points={
        "console_scripts": [
            "nff = nfflr.train:cli",
            "nffd = nfflr.distributed:main",
        ]
    },
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/usnistgov/nfflr",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
