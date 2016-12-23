#!/usr/bin/env python
"""Setup ProtScan."""

from setuptools import setup

__author__ = "Gianluca Corrado"
__copyright__ = "Copyright 2016, Gianluca Corrado"
__license__ = "MIT"
__maintainer__ = "Gianluca Corrado"
__email__ = "gianluca.corrado@unitn.it"
__status__ = "Production"
__version__ = 0.1


setup(
    name='protscan',
    version=__version__,
    author='Gianluca Corrado',
    author_email='gianluca.corrado@unitn.it',
    packages=['protscan',
              'protscan.util'
              ],
    scripts=['bin/protscan'],
    include_package_data=True,
    package_data={},
    license="MIT",
    description="""Protein-RNA target site modelling and prediction.""",
    long_description=open('README.md').read(),
    install_requires=[
        "eden",
        "joblib >= 0.9.4",
        "numpy >= 1.11.2",
        "scipy >= 0.18.1"
    ],
)
