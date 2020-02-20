#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
from setuptools import setup

import re

# load version form _version.py
VERSIONFILE = "incremental_pca/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

# module

setup(name='incremental_pca',
      version=verstr,
      author="Keisuke Fujii",
      author_email="fujii@me.kyoto-u.ac.jp",
      description=("Python library to perform incremental pca with dask"),
      license="BSD 3-clause",
      keywords="machine learning",
      url="http://github.com/fujiisoup/DaskIncrementalPca",
      include_package_data=True,
      ext_modules=[],
      packages=["incremental_pca", ],
      package_dir={'incremental_pca': 'incremental_pca'},
      py_modules=['incremental_pca.__init__'],
      test_suite='incremental_pca/tests',
      install_requires="""
        numpy>=1.11
        dask>=1.00
        """,
      classifiers=['License :: OSI Approved :: BSD License',
                   'Natural Language :: English',
                   'Operating System :: MacOS :: MacOS X',
                   'Operating System :: Microsoft :: Windows',
                   'Operating System :: POSIX :: Linux',
                   'Programming Language :: Python :: 3.6',
                   'Topic :: Scientific/Engineering :: Physics']
      )