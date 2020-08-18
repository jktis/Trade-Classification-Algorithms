# -*- coding: utf-8 -*-
"""
Created on Wed Feb 01 18:40:32 2017

@author: sjurkatis
"""

try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension
    
from Cython.Distutils import build_ext
import numpy

#libraries = ['msvcr90.dll']
ext_modules = [Extension('tradeclassification_c',
                         ['tradeclassification_c.pyx'], 
                         include_dirs = [numpy.get_include()])]
                         
setup(cmdclass = {'build_ext': build_ext}, ext_modules = ext_modules)

#from Cython.Build import cythonize
#setup(ext_modules = cythonize('c_get_inds.pyx' , include_dirs = [numpy.get_include()]))

