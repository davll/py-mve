from setuptools import setup, Extension
import platform, sys, os.path
from os.path import join, isdir
from glob import glob, iglob
from os import environ as ENV

try:
    import numpy
except:
    print('Numpy is required')
    sys.exit(1)

if 'MVE_ROOT' in ENV:
    MVE_ROOT = ENV['MVE_ROOT']
else:
    if platform.system() == 'Darwin':
        MVE_ROOT = '/usr/local/lib/mve'
    elif platform.system() == 'Linux':
        MVE_ROOT = '/opt/mve'
    else:
        raise RuntimeError('Environment Var $MVE_ROOT should be defined')

LIB_PREFIX = join(MVE_ROOT, 'libs')

if not isdir(LIB_PREFIX):
    raise RuntimeError(LIB_PREFIX + ' should be a directory')

def get_include_dirs():
    global LIB_PREFIX
    return [LIB_PREFIX, numpy.get_include()]

def get_library_dirs():
    global LIB_PREFIX
    return [join(LIB_PREFIX, 'mve'), join(LIB_PREFIX, 'util')]

def get_libraries():
    return ['mve', 'mve_util']

def extensions():
    return [Extension('mve.core',
                      sources = glob(join('src', 'core', '*.cpp')),
                      include_dirs = get_include_dirs(),
                      library_dirs = get_library_dirs(),
                      libraries = get_libraries(),
                      language = 'c++',
                      define_macros = [('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
                      )]

setup(
    name = 'mve',
    version = '1.0',
    description = 'Multi-View Environment',
    author = 'David Lin',
    author_email = 'davll.xc@gmail.com',
    packages = ['mve'],
    ext_modules = extensions(),
    install_requires = ['numpy']
)
