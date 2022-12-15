import os
from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


NAME = "uml_imputer"
VERSION = "0.0.1"
AUTHORS = "Bradley T. Martin and Tyler K. Chafin"
AUTHOR_EMAIL = "evobio721@gmail.com"
MAINTAINER = "Bradley T. Martin"
DESCRIPTION = "Python machine learning API to impute missing SNPs"

try:
    import pypandoc

    LONG_DESCRIPTION = open("README.md").read()
except (IOError, ImportError):
    LONG_DESCRIPTION = open("README.md").read()

setup(
    name=NAME,
    version=VERSION,
    author=AUTHORS,
    author_email=AUTHOR_EMAIL,
    maintainer=MAINTAINER,
    maintainer_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/btmartin721/uml_imputer",
    project_urls={
        "Bug Tracker": "https://github.com/btmartin721/uml_imputer/issues"
    },
    keywords=[
        "python",
        "api",
        "impute",
        "imputation",
        "imputer",
        "machine learning",
        "deep learning",
        "neural network",
        "unsupervised",
        "ubp",
        "autoencoder",
        "nlpca",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Operating System :: OS Independent",
        "Natural Language :: English",
    ],
    license="GNU General Public License v3 (GPLv3)",
    packages=find_packages(),
    python_requires=">=3.8.0,<3.9",
    install_requires=[
        "matplotlib",
        "seaborn",
        "jupyterlab",
        "scikit-learn==1.0",
        "tqdm",
        "pandas>=1.2.5,<1.3.0",
        "numpy>=1.20.2,<1.21",
        "scipy>=1.6.2,<1.7",
        "tensorflow-cpu==2.7",
        "keras>=2.7,<2.8",
        "toytree",
        "sklearn-genetic-opt[all]>=0.6.0",
        "importlib-resources>=1.1.0",
        "scikeras>=0.6.0",
        "kaleido",
        "plotly",
        "protobuf>=3.9.2,<4",
        "packaging<22.0",
        "ypy-websocket>=0.5.0,<0.6.0",
    ],
    extras_require={"intel": ["scikit-learn-intelex"], "dev": ["black"], "arm": ["setuptools==57", "mlflow"]},
    package_data={
        "uml_imputer": [
            "example_data/structure_files/*.str",
            "example_data/phylip_files/*.phy",
            "example_data/popmaps/test.popmap",
            "example_data/trees/test*",
        ]
    },
    include_package_data=True,
)
