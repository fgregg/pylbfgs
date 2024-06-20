#!/usr/bin/env python
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from distutils.ccompiler import get_default_compiler
import numpy

try:
    from Cython.Build import cythonize
    use_cython = True
except ImportError:
    use_cython = False

class custom_build_ext(build_ext):
    def finalize_options(self):
        build_ext.finalize_options(self)
        if self.compiler is None:
            compiler = get_default_compiler()
        else:
            compiler = self.compiler

        if compiler == 'msvc':
            include_dirs.append('compat/win32')

include_dirs = ['liblbfgs', numpy.get_include()]

if use_cython:
    ext_modules = cythonize(
        [Extension('lbfgs._lowlevel',
                   ['lbfgs/_lowlevel.pyx', 'liblbfgs/lbfgs.c'],
                   include_dirs=include_dirs)])
else:
    ext_modules = [Extension('lbfgs._lowlevel',
                             ['lbfgs/_lowlevel.c', 'liblbfgs/lbfgs.c'],
                             include_dirs=include_dirs)]


def readme():
    with open('README.rst') as f:
        return f.read()

setup(
    name="PyLBFGS",
    url='https://github.com/dedupeio/pylbfgs',
    version="0.2.0.16",
    description="LBFGS and OWL-QN optimization algorithms",
    author="Lars Buitinck, Forest Gregg",
    author_email="fgregg@gmail.com",
    packages=['lbfgs'],
    install_requires=['numpy>=1.13.1'],
    ext_modules=ext_modules,
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX",
        "Programming Language :: Cython",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development",
    ],
    cmdclass={'build_ext': custom_build_ext},
    long_description=readme(),
)
